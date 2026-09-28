import time

from cpmpy.solvers.solver_interface import (
    SolverInterface,
    SolverStatus,
    ExitStatus,
)
from cpmpy.expressions.variables import (
    _BoolVarImpl,
    _IntVarImpl,
    NegBoolView,
)
from cpmpy.expressions.core import (
    Comparison,
    Operator,
    BoolVal,
)
from cpmpy.expressions.globalconstraints import AllDifferent, Table
from cpmpy.transformations.normalize import toplevel_list
from cpmpy.transformations.decompose_global import decompose_in_tree
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.linearize import canonical_comparison
from cpmpy.transformations.negation import push_down_negation
from cpmpy.transformations.reification import (
    reify_rewrite,
    only_bv_reifies,
    only_implies,
)
from cpmpy.transformations.comparison import only_numexpr_equality
from cpmpy.transformations.get_variables import get_variables


class CPM_maxicp_native(SolverInterface):
    """
    CPMpy interface for MaxiCP through the GraalVM Native Image
    shared-library wrapper.
    """

    @staticmethod
    def supported():
        try:
            from maxicp_native import MaxiCP

            solver = MaxiCP()
            solver.close()

            return True

        except (ImportError, OSError, RuntimeError):
            return False

    @staticmethod
    def version():
        return "native-image"

    def __init__(self, cpm_model=None, subsolver=None):
        if not self.supported():
            raise Exception(
                "MaxiCP native wrapper is not available"
            )

        super().__init__(
            name="maxicp_native",
            cpm_model=None,
        )

        from maxicp_native import MaxiCP

        self.maxicp = MaxiCP()

        self._objective_var = None
        self._objective_is_min = None
        self._decision_vars = []

        if cpm_model is not None:
            original_vars = get_variables(cpm_model.constraints)

            self += cpm_model.constraints

            if cpm_model.objective_ is not None:
                self.objective(
                    cpm_model.objective_,
                    minimize=cpm_model.objective_is_min,
                )

            for cpm_var in original_vars:
                if isinstance(cpm_var, NegBoolView):
                    cpm_var = cpm_var._bv

                native_var = self.solver_var(cpm_var)
                self.maxicp.add_search_var(native_var)

    def __del__(self):
        maxicp = getattr(self, "maxicp", None)

        if maxicp is not None:
            try:
                maxicp.close()
            except Exception:
                pass

    # ============================================================
    # Variables
    # ============================================================

    def solver_var(self, cpm_var):
        if cpm_var in self._varmap:
            return self._varmap[cpm_var]

        # NegBoolView must be handled before _BoolVarImpl.
        if isinstance(cpm_var, NegBoolView):
            native_var = self.maxicp.bool_var()

            self.maxicp.not_(
                native_var,
                self.solver_var(cpm_var._bv),
            )

        elif isinstance(cpm_var, _BoolVarImpl):
            native_var = self.maxicp.bool_var()

        elif isinstance(cpm_var, _IntVarImpl):
            native_var = self.maxicp.int_var(
                int(cpm_var.lb),
                int(cpm_var.ub),
            )

        else:
            raise NotImplementedError(
                f"Unsupported CPMpy variable: {cpm_var!r}"
            )

        self._varmap[cpm_var] = native_var
        self.user_vars.add(cpm_var)

        return native_var

    # ============================================================
    # Objective
    # ============================================================

    def objective(self, expr, minimize):

        # Simple variable objective.
        if isinstance(expr, _IntVarImpl):
            self._objective_var = self.solver_var(expr)
            self._objective_is_min = bool(minimize)
            return self

        # Linear expression objective.
        if isinstance(expr, Operator):
            native_vars, coefficients, offset = self._linear_terms(expr)

            lb, ub = self._expression_bounds(expr)

            objective_var = self.maxicp.int_var(lb, ub)

            # expression + offset = objective
            #
            # expression - objective = -offset
            self.maxicp.linear_eq(
                native_vars + [objective_var],
                coefficients + [-1],
                -int(offset),
            )

            self._objective_var = objective_var
            self._objective_is_min = bool(minimize)

            return self

        raise NotImplementedError(
            f"Unsupported MaxiCP native objective: {expr!r}"
        )

    def _expression_bounds(self, expr):
        """
        Conservative integer bounds for the flattened linear
        expression using CPMpy variable bounds.
        """

        if isinstance(expr, _IntVarImpl):
            return int(expr.lb), int(expr.ub)

        if isinstance(expr, int):
            return int(expr), int(expr)

        if isinstance(expr, Operator) and expr.name == "sum":
            lb = 0
            ub = 0

            for arg in expr.args:
                arg_lb, arg_ub = self._expression_bounds(arg)
                lb += arg_lb
                ub += arg_ub

            return lb, ub

        if isinstance(expr, Operator) and expr.name == "wsum":
            weights, variables = expr.args

            lb = 0
            ub = 0

            for weight, var in zip(weights, variables):
                weight = int(weight)

                var_lb, var_ub = self._expression_bounds(var)

                if weight >= 0:
                    lb += weight * var_lb
                    ub += weight * var_ub
                else:
                    lb += weight * var_ub
                    ub += weight * var_lb

            return lb, ub

        if isinstance(expr, Operator) and expr.name == "sub":
            left, right = expr.args

            left_lb, left_ub = self._expression_bounds(left)
            right_lb, right_ub = self._expression_bounds(right)

            return (
                left_lb - right_ub,
                left_ub - right_lb,
            )

        raise NotImplementedError(
            f"Cannot determine bounds for objective: {expr!r}"
        )

    # ============================================================
    # Transformations
    # ============================================================

    def transform(self, cpm_expr):
        cpm_cons = toplevel_list(cpm_expr)

        cpm_cons = push_down_negation(cpm_cons)

        cpm_cons = decompose_in_tree(
            cpm_cons,
            supported=frozenset({
                "alldifferent",
                "table",
            }),
            supported_reified=frozenset(),
        )

        cpm_cons = flatten_constraint(
            cpm_cons,
            csemap=self._csemap,
        )

        cpm_cons = canonical_comparison(cpm_cons)

        cpm_cons = reify_rewrite(
            cpm_cons,
            supported=frozenset({
                "sum",
                "wsum",
            }),
            csemap=self._csemap,
        )

        cpm_cons = only_numexpr_equality(
            cpm_cons,
            supported=frozenset({
                "sum",
                "wsum",
                "sub",
            }),
            csemap=self._csemap,
        )

        cpm_cons = only_bv_reifies(
            cpm_cons,
            csemap=self._csemap,
        )

        cpm_cons = only_implies(
            cpm_cons,
            csemap=self._csemap,
        )

        return cpm_cons

    # ============================================================
    # Linear expressions
    # ============================================================

    def _linear_terms(self, expr):
        if isinstance(expr, _IntVarImpl):
            return [self.solver_var(expr)], [1], 0

        if isinstance(expr, int):
            return [], [], int(expr)

        if isinstance(expr, Operator) and expr.name == "sum":
            variables = []
            coefficients = []
            offset = 0

            for arg in expr.args:
                if isinstance(arg, int):
                    offset += int(arg)
                else:
                    arg_vars, arg_coefs, arg_offset = self._linear_terms(arg)

                    variables.extend(arg_vars)
                    coefficients.extend(arg_coefs)
                    offset += arg_offset

            return variables, coefficients, offset

        if isinstance(expr, Operator) and expr.name == "wsum":
            weights, cpm_vars = expr.args

            variables = []
            coefficients = []
            offset = 0

            for weight, var in zip(weights, cpm_vars):
                weight = int(weight)

                if isinstance(var, int):
                    offset += weight * int(var)

                elif isinstance(var, _IntVarImpl):
                    variables.append(self.solver_var(var))
                    coefficients.append(weight)

                else:
                    sub_vars, sub_coefs, sub_offset = self._linear_terms(var)

                    variables.extend(sub_vars)
                    coefficients.extend(
                        weight * coef
                        for coef in sub_coefs
                    )
                    offset += weight * sub_offset

            return variables, coefficients, offset

        if isinstance(expr, Operator) and expr.name == "sub":
            left, right = expr.args

            left_vars, left_coefs, left_offset = self._linear_terms(left)
            right_vars, right_coefs, right_offset = self._linear_terms(right)

            return (
                left_vars + right_vars,
                left_coefs + [-c for c in right_coefs],
                left_offset - right_offset,
            )

        raise NotImplementedError(
            f"Unsupported native MaxiCP numeric expression: {expr!r}"
        )

    # ============================================================
    # Reification
    # ============================================================

    def _new_bool(self):
        return self.maxicp.bool_var()

    def _native_not(self, value):
        result = self._new_bool()

        self.maxicp.not_(
            result,
            value,
        )

        return result

    def _native_and(self, values):
        result = self._new_bool()

        self.maxicp.and_(
            result,
            list(values),
        )

        return result

    def _native_or(self, values):
        result = self._new_bool()

        self.maxicp.or_(
            result,
            list(values),
        )

        return result

    def _reify_comparison(self, con):
        lhs, rhs = con.args
        op = con.name

        if not isinstance(rhs, int):
            raise NotImplementedError(
                f"Expected constant RHS after transformation, got: {con!r}"
            )

        variables, coefficients, offset = self._linear_terms(lhs)

        rhs = int(rhs) - offset

        if not variables:
            truth = {
                "==": 0 == rhs,
                "!=": 0 != rhs,
                "<=": 0 <= rhs,
                "<": 0 < rhs,
                ">=": 0 >= rhs,
                ">": 0 > rhs,
            }[op]

            result = self._new_bool()
            self.maxicp.eq(result, int(truth))
            return result

        if op == "==":
            result = self._new_bool()

            self.maxicp.is_linear_eq(
                result,
                variables,
                coefficients,
                rhs,
            )

            return result

        if op == "!=":
            equality = self._new_bool()

            self.maxicp.is_linear_eq(
                equality,
                variables,
                coefficients,
                rhs,
            )

            return self._native_not(equality)

        if op == "<=":
            result = self._new_bool()

            self.maxicp.is_linear_le(
                result,
                variables,
                coefficients,
                rhs,
            )

            return result

        if op == "<":
            result = self._new_bool()

            self.maxicp.is_linear_le(
                result,
                variables,
                coefficients,
                rhs - 1,
            )

            return result

        if op == ">=":
            less = self._new_bool()

            self.maxicp.is_linear_le(
                less,
                variables,
                coefficients,
                rhs - 1,
            )

            return self._native_not(less)

        if op == ">":
            less = self._new_bool()

            self.maxicp.is_linear_le(
                less,
                variables,
                coefficients,
                rhs,
            )

            return self._native_not(less)

        raise NotImplementedError(
            f"Unsupported reified comparison: {con!r}"
        )

    def _bool_expr(self, expr):
        if isinstance(expr, NegBoolView):
            return self.solver_var(expr)

        if isinstance(expr, _BoolVarImpl):
            return self.solver_var(expr)

        if isinstance(expr, BoolVal):
            result = self._new_bool()
            self.maxicp.eq(result, int(bool(expr.value())))
            return result

        if isinstance(expr, Comparison):
            return self._reify_comparison(expr)

        if isinstance(expr, Operator):
            if expr.name == "or":
                return self._native_or(
                    self._bool_expr(arg)
                    for arg in expr.args
                )

            if expr.name == "and":
                return self._native_and(
                    self._bool_expr(arg)
                    for arg in expr.args
                )

        raise NotImplementedError(
            f"Unsupported Boolean expression: {expr!r}"
        )

    # ============================================================
    # Posting
    # ============================================================

    def _post_comparison(self, con):
        lhs, rhs = con.args
        op = con.name

        if not isinstance(rhs, int):
            raise NotImplementedError(
                f"Expected constant RHS after transformation, got: {con!r}"
            )

        variables, coefficients, offset = self._linear_terms(lhs)

        rhs = int(rhs) - offset

        # Constant-only comparison.
        if not variables:
            satisfied = {
                "==": 0 == rhs,
                "!=": 0 != rhs,
                "<=": 0 <= rhs,
                "<": 0 < rhs,
                ">=": 0 >= rhs,
                ">": 0 > rhs,
            }[op]

            if not satisfied:
                b = self.maxicp.bool_var()
                self.maxicp.eq(b, 0)
                self.maxicp.eq(b, 1)

            return

        if op == "!=":
            equality = self._new_bool()

            self.maxicp.is_linear_eq(
                equality,
                variables,
                coefficients,
                rhs,
            )

            self.maxicp.eq(
                equality,
                0,
            )

            return

        if op == "<":
            op = "<="
            rhs -= 1

        elif op == ">":
            op = ">="
            rhs += 1

        if op == "==":
            self.maxicp.linear_eq(
                variables,
                coefficients,
                rhs,
            )

        elif op == "<=":
            self.maxicp.linear_le(
                variables,
                coefficients,
                rhs,
            )

        elif op == ">=":
            self.maxicp.linear_ge(
                variables,
                coefficients,
                rhs,
            )

        else:
            raise NotImplementedError(
                f"Unsupported comparison operator: {con!r}"
            )

    def add(self, cpm_expr):
        transformed = self.transform(cpm_expr)

        for con in transformed:

            # Constant Boolean.
            if isinstance(con, BoolVal):
                if bool(con.value()):
                    continue

                b = self.maxicp.bool_var()
                self.maxicp.eq(b, 0)
                self.maxicp.eq(b, 1)
                continue

            # ~b
            if isinstance(con, NegBoolView):
                native_bool = self.solver_var(con._bv)
                self.maxicp.eq(native_bool, 0)
                continue

            # b
            if isinstance(con, _BoolVarImpl):
                native_bool = self.solver_var(con)
                self.maxicp.eq(native_bool, 1)
                continue

            # Numeric comparison.
            if isinstance(con, Comparison):
                self._post_comparison(con)
                continue

            # Boolean AND / OR.
            if isinstance(con, Operator) and con.name in {"or", "and"}:
                native_bool = self._bool_expr(con)

                self.maxicp.eq(
                    native_bool,
                    1,
                )

                continue

            # Implication.
            if isinstance(con, Operator) and con.name == "->":
                antecedent, consequent = con.args

                native_antecedent = self._bool_expr(
                    antecedent
                )

                if isinstance(consequent, Comparison):
                    native_consequent = self._reify_comparison(
                        consequent
                    )
                else:
                    native_consequent = self._bool_expr(
                        consequent
                    )

                # a -> b  <=>  a <= b
                self.maxicp.le_var(
                    native_antecedent,
                    native_consequent,
                )

                continue

            # AllDifferent.
            if isinstance(con, AllDifferent):
                native_vars = [
                    self.solver_var(var)
                    for var in con.args
                ]

                self.maxicp.all_different(
                    native_vars
                )

                continue

            # Global positive Table constraint.
            if isinstance(con, Table):
                array, table = con.args

                native_vars = [
                    self.solver_var(var)
                    for var in array
                ]

                self.maxicp.table(
                    native_vars,
                    table.tolist(),
                )

                continue

            raise NotImplementedError(
                f"Unsupported transformed constraint: {con!r}"
            )

        return self

    # ============================================================
    # Solve
    # ============================================================

    def solve(self, time_limit=None, **kwargs):
        if kwargs:
            raise NotImplementedError(
                f"Unsupported MaxiCP native solve arguments: {kwargs}"
            )

        start = time.time()

        if self._objective_var is None:

            if time_limit is None:
                result = (
                    self.maxicp.RESULT_COMPLETE
                    if self.maxicp.solve()
                    else self.maxicp.RESULT_UNSAT
                )

            else:
                result = self.maxicp.solve_limit(
                    max(0, int(float(time_limit) * 1000))
                )

        else:

            if self._objective_is_min:
                result = self.maxicp.minimize(
                    self._objective_var,
                    time_limit=time_limit,
                )
            else:
                result = self.maxicp.maximize(
                    self._objective_var,
                    time_limit=time_limit,
                )

        runtime = time.time() - start

        has_solution = result in (
            self.maxicp.RESULT_COMPLETE,
            self.maxicp.RESULT_TIMEOUT_WITH_SOLUTION,
        )

        if result == self.maxicp.RESULT_UNSAT:
            exitstatus = ExitStatus.UNSATISFIABLE

        elif result == self.maxicp.RESULT_COMPLETE:
            if self._objective_var is None:
                exitstatus = ExitStatus.FEASIBLE
            else:
                exitstatus = ExitStatus.OPTIMAL

        elif result == self.maxicp.RESULT_TIMEOUT_WITH_SOLUTION:
            exitstatus = ExitStatus.FEASIBLE

        elif result == self.maxicp.RESULT_TIMEOUT_NO_SOLUTION:
            exitstatus = ExitStatus.UNKNOWN

        else:
            raise RuntimeError(
                f"Unexpected MaxiCP native result code: {result}"
            )

        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.exitstatus = exitstatus
        self.cpm_status.runtime = runtime

        if has_solution:
            for cpm_var in self.user_vars:
                if isinstance(cpm_var, NegBoolView):
                    continue

                native_var = self.solver_var(cpm_var)
                cpm_var._value = native_var.value()

            if self._objective_var is not None:
                self.objective_value_ = self._objective_var.value()
            else:
                self.objective_value_ = None

        else:
            for cpm_var in self.user_vars:
                if isinstance(cpm_var, NegBoolView):
                    continue

                cpm_var._value = None

            self.objective_value_ = None

        return has_solution

    __add__ = add