import time

from cpmpy.solvers.solver_interface import SolverInterface, SolverStatus, ExitStatus
from cpmpy.expressions.variables import (
    _BoolVarImpl,
    _IntVarImpl,
    NegBoolView,
)
from cpmpy.expressions.core import Comparison, Operator, BoolVal
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


class CPM_minicp_native(SolverInterface):
    """
    CPMpy interface for MiniCP through the GraalVM Native Image
    shared-library wrapper.
    """

    @staticmethod
    def supported():
        try:
            from minicp_native import MiniCP

            solver = MiniCP()
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
                "MiniCP native wrapper is not available"
            )

        super().__init__(
            name="minicp_native",
            cpm_model=None,
        )

        from minicp_native import MiniCP

        self.minicp = MiniCP()

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
                self.minicp.add_search_var(native_var)

    def __del__(self):
        minicp = getattr(self, "minicp", None)

        if minicp is not None:
            try:
                minicp.close()
            except Exception:
                pass

    def solver_var(self, cpm_var):
        if cpm_var in self._varmap:
            return self._varmap[cpm_var]

        # Negated Boolean view must refer to the same underlying Boolean.
        # Do this before _BoolVarImpl because NegBoolView is also Boolean-like.
        if isinstance(cpm_var, NegBoolView):
            native_var = self.minicp.not_(
                self.solver_var(cpm_var._bv)
            )

        elif isinstance(cpm_var, _BoolVarImpl):
            native_var = self.minicp.boolvar()

        elif isinstance(cpm_var, _IntVarImpl):
            native_var = self.minicp.intvar(
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

    def _linear_bounds(self, variables, coefficients, offset=0):
        lb = int(offset)
        ub = int(offset)

        for var, coef in zip(variables, coefficients):
            coef = int(coef)

            if coef >= 0:
                lb += coef * int(var.lb)
                ub += coef * int(var.ub)
            else:
                lb += coef * int(var.ub)
                ub += coef * int(var.lb)

        return lb, ub

    def objective(self, expr, minimize):

        # Simple variable objective
        if isinstance(expr, _IntVarImpl):
            self._objective_var = self.solver_var(expr)
            self._objective_is_min = bool(minimize)
            return self

        # Linear expression objective
        if isinstance(expr, Operator):
            native_vars, coefficients, offset = self._linear_terms(expr)

            lb = int(offset)
            ub = int(offset)

            for native_var, coef in zip(native_vars, coefficients):
                coef = int(coef)

                if coef >= 0:
                    lb += coef * native_var.lb
                    ub += coef * native_var.ub
                else:
                    lb += coef * native_var.ub
                    ub += coef * native_var.lb

            objective_var = self.minicp.intvar(lb, ub)

            # objective_var = linear expression + offset
            #
            # linear expression - objective_var = -offset
            self.minicp.linear(
                native_vars + [objective_var],
                coefficients + [-1],
                "==",
                -int(offset),
            )

            self._objective_var = objective_var
            self._objective_is_min = bool(minimize)

            return self

        raise NotImplementedError(
            f"Unsupported MiniCP native objective: {expr!r}"
        )

    def transform(self, cpm_expr):
        """
        Transform arbitrary CPMpy expressions into the small constraint
        language supported by the native MiniCP bridge.
        """

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

    def _linear_terms(self, expr):
        """
        Convert a flattened CPMpy numeric expression into:
            ([native variables], [coefficients], constant_offset)
        """

        # Plain CPMpy variable
        if isinstance(expr, _IntVarImpl):
            return [self.solver_var(expr)], [1], 0

        # sum(x, y, ...)
        if isinstance(expr, Operator) and expr.name == "sum":
            variables = []
            coefficients = []
            offset = 0

            for arg in expr.args:
                if isinstance(arg, int):
                    offset += int(arg)
                else:
                    variables.append(self.solver_var(arg))
                    coefficients.append(1)

            return variables, coefficients, offset

        # wsum([weights], [variables])
        if isinstance(expr, Operator) and expr.name == "wsum":
            weights, cpm_vars = expr.args

            variables = []
            coefficients = []
            offset = 0

            for weight, var in zip(weights, cpm_vars):
                weight = int(weight)

                if isinstance(var, int):
                    offset += weight * int(var)
                else:
                    variables.append(self.solver_var(var))
                    coefficients.append(weight)

            return variables, coefficients, offset

        # sub(a,b)
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
            f"Unsupported native MiniCP numeric expression: {expr!r}"
        )

    def _reify_comparison(self, con):
        """
        Return a native MiniCP BoolVar whose value is equivalent
        to the normalized CPMpy comparison.
        """
        lhs, rhs = con.args
        op = con.name

        if not isinstance(rhs, int):
            raise NotImplementedError(
                f"Expected constant RHS after transformation, got: {con!r}"
            )

        variables, coefficients, offset = self._linear_terms(lhs)
        rhs = int(rhs) - offset

        if op == "==":
            return self.minicp.is_linear_equal(
                variables, coefficients, rhs
            )

        if op == "!=":
            equality = self.minicp.is_linear_equal(
                variables, coefficients, rhs
            )
            return self.minicp.not_(equality)

        if op == "<=":
            return self.minicp.is_linear_less_or_equal(
                variables, coefficients, rhs
            )

        if op == "<":
            return self.minicp.is_linear_less_or_equal(
                variables, coefficients, rhs - 1
            )

        if op == ">=":
            # E >= rhs  <=>  NOT(E <= rhs - 1)
            less = self.minicp.is_linear_less_or_equal(
                variables, coefficients, rhs - 1
            )
            return self.minicp.not_(less)

        if op == ">":
            # E > rhs  <=>  NOT(E <= rhs)
            less = self.minicp.is_linear_less_or_equal(
                variables, coefficients, rhs
            )
            return self.minicp.not_(less)

        raise NotImplementedError(
            f"Unsupported reified comparison: {con!r}"
        )

    def _bool_expr(self, expr):
        """
        Convert a normalized CPMpy Boolean expression into
        a native MiniCP BoolVar.
        """

        if isinstance(expr, NegBoolView):
            return self.minicp.not_(
                self.solver_var(expr._bv)
            )

        if isinstance(expr, _BoolVarImpl):
            return self.solver_var(expr)

        if isinstance(expr, Operator):

            if expr.name == "or":
                return self.minicp.or_([
                    self._bool_expr(arg)
                    for arg in expr.args
                ])

            if expr.name == "and":
                return self.minicp.and_([
                    self._bool_expr(arg)
                    for arg in expr.args
                ])

        raise NotImplementedError(
            f"Unsupported Boolean expression: {expr!r}"
        )

    def _post_comparison(self, con):
        lhs, rhs = con.args
        op = con.name

        if not isinstance(rhs, int):
            raise NotImplementedError(
                f"Expected constant RHS after transformation, got: {con!r}"
            )

        variables, coefficients, offset = self._linear_terms(lhs)

        # lhs + offset OP rhs
        # becomes lhs OP (rhs - offset)
        rhs = int(rhs) - offset

        # Top-level linear disequality:
        # post MiniCP's native disequality constraint directly
        # instead of building a reified equality and negating it.
        if op == "!=":
            self.minicp.linear(
                variables,
                coefficients,
                "!=",
                rhs,
            )
            return

        if op == "<":
            op = "<="
            rhs -= 1

        elif op == ">":
            op = ">="
            rhs += 1

        if op not in {"==", "<=", ">="}:
            raise NotImplementedError(
                f"Unsupported comparison operator: {con!r}"
            )

        self.minicp.linear(
            variables,
            coefficients,
            op,
            rhs,
        )

    def add(self, cpm_expr):
        transformed = self.transform(cpm_expr)

        for con in transformed:

            # Constant Boolean
            if isinstance(con, BoolVal):
                if bool(con.value()):
                    # True constraint: no-op, trivially satisfied.
                    continue

                # False constraint: make the native model inconsistent.
                b = self.minicp.boolvar()
                self.minicp.eq(b, 0)
                self.minicp.eq(b, 1)
                continue

            # Negated Boolean variable: ~b
            if isinstance(con, NegBoolView):
                native_bool = self.solver_var(con._bv)
                self.minicp.eq(native_bool, 0)
                continue

            # Bare Boolean variable: b
            if isinstance(con, _BoolVarImpl):
                native_bool = self.solver_var(con)
                self.minicp.eq(native_bool, 1)
                continue

            # Ordinary top-level numeric comparison
            if isinstance(con, Comparison):
                self._post_comparison(con)
                continue

            # Top-level Boolean OR / AND
            if isinstance(con, Operator) and con.name in {"or", "and"}:
                native_bool = self._bool_expr(con)
                self.minicp.eq(native_bool, 1)
                continue

            # Boolean implication
            if isinstance(con, Operator) and con.name == "->":
                antecedent, consequent = con.args

                native_antecedent = self._bool_expr(antecedent)

                if isinstance(consequent, Comparison):
                    native_consequent = self._reify_comparison(consequent)
                else:
                    native_consequent = self._bool_expr(consequent)

                # a -> b  <=>  a <= b
                self.minicp.le_var(
                    native_antecedent,
                    native_consequent,
                )
                continue

            # Global AllDifferent constraint
            if isinstance(con, AllDifferent):
                native_vars = [
                    self.solver_var(var)
                    for var in con.args
                ]

                self.minicp.all_different(native_vars)
                continue

            # Global positive Table constraint
            if isinstance(con, Table):
                array, table = con.args

                native_vars = [
                    self.solver_var(var)
                    for var in array
                ]

                self.minicp.table(
                    native_vars,
                    table.tolist(),
                )
                continue

            raise NotImplementedError(
                f"Unsupported transformed constraint: {con!r}"
            )

        return self

    def solve(self, time_limit=None, **kwargs):
        if kwargs:
            raise NotImplementedError(
                f"Unsupported MiniCP native solve arguments: {kwargs}"
            )

        start = time.time()

        if time_limit is None:
            if self._objective_var is None:
                result = 1 if self.minicp.solve() else 0
            else:
                result = 1 if self.minicp.optimize(
                    self._objective_var,
                    minimize=self._objective_is_min,
                ) else 0

        else:
            time_limit_ms = max(0, int(float(time_limit) * 1000))

            if self._objective_var is None:
                result = self.minicp.solve_limit(
                    time_limit_ms
                )
            else:
                result = self.minicp.optimize_limit(
                    self._objective_var,
                    minimize=self._objective_is_min,
                    time_limit_ms=time_limit_ms,
                )

        runtime = time.time() - start

        has_solution = result in (1, 2)

        if result == 0:
            exitstatus = ExitStatus.UNSATISFIABLE

        elif result == 1:
            if self._objective_var is None:
                exitstatus = ExitStatus.FEASIBLE
            else:
                exitstatus = ExitStatus.OPTIMAL

        elif result == 2:
            # Search stopped by time limit, but a solution/incumbent exists.
            exitstatus = ExitStatus.FEASIBLE

        elif result == 3:
            # Search stopped before finding a solution.
            exitstatus = ExitStatus.UNKNOWN

        else:
            raise RuntimeError(
                f"Unexpected MiniCP native result code: {result}"
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