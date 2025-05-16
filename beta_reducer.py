import copy
from typing import Set, Tuple, List

from typed_lambda_parser import (
    Expression, Variable, Abstraction, Application,
    Type, # TypeVariable, FunctionType are implicitly handled via Type
    tokenize, Parser, ParserSyntaxError
)

# --- Free Variable Calculation ---

def get_free_variables(expr: Expression) -> Set[str]:
    """Computes the set of free variable names in an expression."""
    if isinstance(expr, Variable):
        return {expr.name}
    elif isinstance(expr, Abstraction):
        return get_free_variables(expr.body) - {expr.param_name}
    elif isinstance(expr, Application):
        return get_free_variables(expr.func) | get_free_variables(expr.arg)
    else:
        # Should not happen with valid Expression types
        raise TypeError(f"Unknown expression type: {type(expr)}")

# --- Alpha Conversion Support (Fresh Variable Generation) ---

def generate_fresh_variable(base_name: str, forbidden_vars: Set[str]) -> str:
    """Generates a fresh variable name based on base_name, avoiding names in forbidden_vars."""
    if base_name not in forbidden_vars:
        return base_name
    
    new_name = base_name + "'"
    while new_name in forbidden_vars:
        new_name += "'"
    return new_name

# --- Substitution ---

def substitute(expr: Expression, var_name_to_replace: str, replacement_expr: Expression) -> Expression:
    """
    Substitutes all free occurrences of var_name_to_replace in expr with replacement_expr.
    Performs alpha-conversion to avoid variable capture.
    """
    if isinstance(expr, Variable):
        if expr.name == var_name_to_replace:
            return copy.deepcopy(replacement_expr)
        return expr
    elif isinstance(expr, Application):
        new_func = substitute(expr.func, var_name_to_replace, replacement_expr)
        new_arg = substitute(expr.arg, var_name_to_replace, replacement_expr)
        # Only create a new object if something changed to preserve object identity where possible
        if new_func is not expr.func or new_arg is not expr.arg:
            return Application(new_func, new_arg)
        return expr
    elif isinstance(expr, Abstraction):
        # If the bound variable is the one we are replacing, then var_name_to_replace is shadowed.
        # No substitution occurs in the body for this var_name_to_replace.
        if expr.param_name == var_name_to_replace:
            return expr

        # Check for potential capture of free variables in replacement_expr by expr.param_name
        fv_replacement = get_free_variables(replacement_expr)
        if expr.param_name in fv_replacement:
            # Alpha-conversion: rename expr.param_name to avoid capture
            forbidden_for_new_param = get_free_variables(expr.body) | fv_replacement
            new_param_name = generate_fresh_variable(expr.param_name, forbidden_for_new_param)
            
            # Rename expr.param_name to new_param_name in the body
            # This substitute call renames the original param_name in the body
            renamed_body = substitute(expr.body, expr.param_name, Variable(new_param_name))
            
            # Now, substitute var_name_to_replace in the alpha-converted body
            new_body_after_substitution = substitute(renamed_body, var_name_to_replace, replacement_expr)
            return Abstraction(new_param_name, expr.param_type, new_body_after_substitution)
        else:
            # No capture, proceed with substitution in the body
            new_body = substitute(expr.body, var_name_to_replace, replacement_expr)
            if new_body is not expr.body:
                 return Abstraction(expr.param_name, expr.param_type, new_body)
            return expr # No change in body
    else:
        raise TypeError(f"Unknown expression type: {type(expr)}")

# --- Beta Reduction ---

def beta_reduce_single_step(expr: Expression) -> Tuple[Expression, bool]:
    """
    Performs a single step of beta-reduction (leftmost-outermost).
    Returns (new_expression, True) if a reduction occurred, (expression, False) otherwise.
    """
    if isinstance(expr, Application):
        # Case 1: The application itself is a redex (λx.M) N
        if isinstance(expr.func, Abstraction):
            # This is a redex: (λparam.body) arg
            # Perform substitution: body[param := arg]
            result = substitute(expr.func.body, expr.func.param_name, expr.arg)
            return result, True
        
        # Case 2: Try to reduce the function part
        reduced_func, changed_func = beta_reduce_single_step(expr.func)
        if changed_func:
            return Application(reduced_func, expr.arg), True
            
        # Case 3: Try to reduce the argument part
        reduced_arg, changed_arg = beta_reduce_single_step(expr.arg)
        if changed_arg:
            return Application(expr.func, reduced_arg), True
            
        # No reduction possible in this application
        return expr, False
        
    elif isinstance(expr, Abstraction):
        # Try to reduce the body of the abstraction
        reduced_body, changed_body = beta_reduce_single_step(expr.body)
        if changed_body:
            return Abstraction(expr.param_name, expr.param_type, reduced_body), True
        # No reduction possible in the body
        return expr, False
        
    elif isinstance(expr, Variable):
        # Variables are normal forms, no reduction possible
        return expr, False
        
    else:
        raise TypeError(f"Unknown expression type: {type(expr)}")

def reduce_to_normal_form(expr: Expression, max_steps: int = 100) -> Tuple[Expression, List[Expression]]:
    """
    Reduces an expression to its normal form by repeatedly applying beta_reduce_single_step.
    Returns the normal form and the history of reduction steps.
    Stops if max_steps are reached.
    """
    current_expr = expr
    history = [copy.deepcopy(current_expr)] # Store initial expression

    for step_count in range(max_steps):
        next_expr, changed = beta_reduce_single_step(current_expr)
        if not changed:
            break  # Normal form reached
        current_expr = next_expr
        history.append(copy.deepcopy(current_expr))
    else:
        if changed: # True if loop finished due to max_steps and last step was a change
            print(f"Warning: Reduction stopped after {max_steps} steps. May not be in normal form.")
            
    return current_expr, history

# --- Example Usage ---
if __name__ == '__main__':
    test_cases = [
        ("(λx:T. x) y", "Simple identity application"),
        ("(λf:A->A. λx:A. f x) (λy:A. y) z", "Identity applied to identity, then to z"),
        ("(λf:A->A. λx:A. f (f x)) (λy:A. y) z", "Twice combinator (S S K like) applied to identity, then to z"),
        ("(λx:T. λy:U. x) p q", "K combinator like: (λx.λy.x) p q -> p"),
        ("(λx:((A->B)->A->B). λy:(A->B). λz:A. (x y) z) (λf:A->B. λa:A. f a) (λb:A.b) c", "Complex application"),
        # Test capture avoidance
        # (λx. (λy. x y)) y  -- where the argument y is the same name as the inner binder
        # To represent this, the types need to be consistent. Let T be the type of y.
        ("(λx:T->T. (λy:T. x y)) (λz:T. z)", "Capture avoidance: (λx.(λy. x y)) I -> λy. I y"),
        ("(λx:T. λy:T. x) y", "Capture: (λx.λy.x)y -> λy'.y (if outer y is free and named 'y')"),
        # For the above, let's make the argument distinct to avoid initial confusion with parser
        # (λf: (T->U) -> (T->U). λg: T->U. λa: T. (f g) a) (λh: T->U. h) (λx:T.x) y
        # This is a bit complex to write directly, let's use a simpler capture case:
        # (λx:SomeType. (λy:ParamType. x)) y_arg
        # If y_arg is Variable("y") and ParamType is type of "y"
        # (λx:T. (λy:T. x)) y_var_named_y
        # This requires y_var_named_y to be parsed as a variable.
        # Let's parse: (λf:A. (λg:A. f)) g_outer
        # where g_outer is a variable named 'g' of type A.
        # The string would be: "(λf:A. (λg:A. f)) g"
        ("(λf:A. (λg:A. f)) g_var", "Capture avoidance: (λf.(λg.f))g -> λg'.g_var"),
    ]

    # A helper to parse, assuming T, A, B etc. are base types
    def parse_expr_str(expr_str: str) -> Expression:
        tokens = tokenize(expr_str)
        parser = Parser(tokens)
        return parser.parse()

    for expr_str, description in test_cases:
        print(f"--- Testing: {description} ---")
        print(f"Original: {expr_str}")
        try:
            ast = parse_expr_str(expr_str)
            print(f"Parsed AST: {ast}")
            
            normal_form, history = reduce_to_normal_form(ast)
            
            if len(history) > 1:
                print("Reduction steps:")
                for i, step_expr in enumerate(history):
                    if i == 0:
                        print(f"  Start: {step_expr}")
                    else:
                        print(f"  Step {i}: {step_expr}")
            else:
                print("Expression is already in normal form or no reduction occurred.")
            
            print(f"Normal form: {normal_form}")
            
        except ParserSyntaxError as e:
            print(f"Syntax Error during parsing: {e}")
        except Exception as e:
            print(f"An error occurred during reduction: {e}")
            import traceback
            traceback.print_exc()
        print("-" * 40)

    # Example that might lead to capture if not handled:
    # (λx. λy. x) y  -- if the argument 'y' is free.
    # Let's assume 'y_free' is a variable.
    # The expression (λx:TypeY -> TypeY. (λy:TypeY. x y)) y_free
    # where y_free is a variable of type TypeY.
    # String: "(λx:TY->TY. (λy:TY. x y)) y"
    # This should reduce to (λy':TY. y y') if y_free is named 'y'.
    print("--- Testing specific capture case ---")
    # (λx:T. (λy:T. x)) y_named_arg
    # where y_named_arg is a variable named 'y' of type T
    # String: "(λx:T. (λy:T. x)) y"
    expr_str_capture = "(λf:TypeA. (λg:TypeA. f)) g" # g is free variable here
    print(f"Original: {expr_str_capture}")
    try:
        # Manually construct AST to ensure g is Variable("g")
        # λf:TypeA. (λg:TypeA. f)
        # func_part = Abstraction("f", TypeVariable("TypeA"),
        #                         Abstraction("g", TypeVariable("TypeA"), Variable("f")))
        # arg_part = Variable("g")
        # ast_capture = Application(func_part, arg_part)
        # Or parse it, assuming parser handles 'g' as a variable.
        ast_capture = parse_expr_str(expr_str_capture)

        print(f"Parsed AST for capture test: {ast_capture}")
        normal_form, history = reduce_to_normal_form(ast_capture)
        if len(history) > 1:
            print("Reduction steps:")
            for i, step_expr in enumerate(history):
                print(f"  {'Start:' if i == 0 else 'Step '+str(i)+':'} {step_expr}")
        print(f"Normal form: {normal_form}")
    except Exception as e:
        print(f"Error in capture test: {e}")
        import traceback
        traceback.print_exc()
    print("-" * 40)

