//! Tiny safe evaluator for `f(x, y)` expressions.
//!
//! This is the "controlled freedom" of the landscape tool: the model can
//! describe any surface as math, but it is *not* a code sandbox — there
//! are no side effects, no I/O, no recursion, and evaluation cost is
//! bounded by the mesh size the host chooses.

/// An parsed expression over the variables `x` and `y`.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Const(f64),
    Var(Var),
    Neg(Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Call(Func, Vec<Expr>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Var {
    X,
    Y,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Func {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,
    Sqrt,
    Abs,
    Exp,
    Ln,
    Log2,
    Log10,
    Floor,
    Ceil,
    Round,
    Min,
    Max,
}

impl Func {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "sin" => Self::Sin,
            "cos" => Self::Cos,
            "tan" => Self::Tan,
            "asin" => Self::Asin,
            "acos" => Self::Acos,
            "atan" => Self::Atan,
            "atan2" => Self::Atan2,
            "sqrt" => Self::Sqrt,
            "abs" => Self::Abs,
            "exp" => Self::Exp,
            "ln" | "log" => Self::Ln,
            "log2" => Self::Log2,
            "log10" => Self::Log10,
            "floor" => Self::Floor,
            "ceil" => Self::Ceil,
            "round" => Self::Round,
            "min" => Self::Min,
            "max" => Self::Max,
            _ => return None,
        })
    }

    fn arity(self) -> usize {
        match self {
            Self::Atan2 | Self::Min | Self::Max => 2,
            _ => 1,
        }
    }

    fn call(self, args: &[f64]) -> f64 {
        let a = args.first().copied().unwrap_or(f64::NAN);
        match self {
            Self::Sin => a.sin(),
            Self::Cos => a.cos(),
            Self::Tan => a.tan(),
            Self::Asin => a.asin(),
            Self::Acos => a.acos(),
            Self::Atan => a.atan(),
            Self::Atan2 => a.atan2(args.get(1).copied().unwrap_or(f64::NAN)),
            Self::Sqrt => a.sqrt(),
            Self::Abs => a.abs(),
            Self::Exp => a.exp(),
            Self::Ln => a.ln(),
            Self::Log2 => a.log2(),
            Self::Log10 => a.log10(),
            Self::Floor => a.floor(),
            Self::Ceil => a.ceil(),
            Self::Round => a.round(),
            Self::Min => a.min(args.get(1).copied().unwrap_or(f64::NAN)),
            Self::Max => a.max(args.get(1).copied().unwrap_or(f64::NAN)),
        }
    }
}

impl Expr {
    pub fn parse(source: &str) -> Result<Self, String> {
        let toks = tokenize(source)?;
        let mut p = Parser { toks, pos: 0 };
        let expr = p.expr()?;
        if p.pos != p.toks.len() {
            return Err(format!(
                "unexpected trailing input at position {}",
                p.pos
            ));
        }
        Ok(expr)
    }

    pub fn eval(&self, x: f64, y: f64) -> f64 {
        match self {
            Self::Const(c) => *c,
            Self::Var(Var::X) => x,
            Self::Var(Var::Y) => y,
            Self::Neg(e) => -e.eval(x, y),
            Self::Binary(op, l, r) => {
                let (l, r) = (l.eval(x, y), r.eval(x, y));
                match op {
                    BinOp::Add => l + r,
                    BinOp::Sub => l - r,
                    BinOp::Mul => l * r,
                    BinOp::Div => l / r,
                    BinOp::Rem => l % r,
                    BinOp::Pow => l.powf(r),
                }
            }
            Self::Call(f, args) => {
                let vals: Vec<f64> = args.iter().map(|a| a.eval(x, y)).collect();
                f.call(&vals)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Num(f64),
    Ident(String),
    Op(char),
    LParen,
    RParen,
    Comma,
}

fn tokenize(src: &str) -> Result<Vec<Token>, String> {
    let chars: Vec<char> = src.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() || c == '.' {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            // Exponent: only when it is clearly a numeric exponent
            // ("1e3", "1e-2"), not the constant e ("2*e", "2e").
            if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                let next = chars.get(i + 1);
                let has_digits = matches!(next, Some(d) if d.is_ascii_digit())
                    || matches!(next, Some('+') | Some('-'))
                        && matches!(chars.get(i + 2), Some(d) if d.is_ascii_digit());
                if has_digits {
                    i += 1;
                    if i < chars.len() && (chars[i] == '+' || chars[i] == '-') {
                        i += 1;
                    }
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            let text: String = chars[start..i].iter().collect();
            let n: f64 = text
                .parse()
                .map_err(|_| format!("invalid number '{text}'"))?;
            toks.push(Token::Num(n));
        } else if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            toks.push(Token::Ident(chars[start..i].iter().collect()));
        } else {
            match c {
                '+' | '-' | '*' | '/' | '%' | '^' => toks.push(Token::Op(c)),
                '(' => toks.push(Token::LParen),
                ')' => toks.push(Token::RParen),
                ',' => toks.push(Token::Comma),
                _ => return Err(format!("unexpected character '{c}'")),
            }
            i += 1;
        }
    }
    Ok(toks)
}

/// Recursive-descent parser. Precedence, loosest to tightest:
/// `+ -` → `* / %` → unary sign → `^` (right-associative) → atoms.
struct Parser {
    toks: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.toks.get(self.pos)
    }

    fn next(&mut self) -> Option<Token> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn expr(&mut self) -> Result<Expr, String> {
        let mut lhs = self.term()?;
        while let Some(Token::Op('+' | '-')) = self.peek() {
            let op = match self.next() {
                Some(Token::Op('+')) => BinOp::Add,
                Some(Token::Op('-')) => BinOp::Sub,
                _ => unreachable!(),
            };
            let rhs = self.term()?;
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn term(&mut self) -> Result<Expr, String> {
        let mut lhs = self.factor()?;
        while let Some(Token::Op('*' | '/' | '%')) = self.peek() {
            let op = match self.next() {
                Some(Token::Op('*')) => BinOp::Mul,
                Some(Token::Op('/')) => BinOp::Div,
                Some(Token::Op('%')) => BinOp::Rem,
                _ => unreachable!(),
            };
            let rhs = self.factor()?;
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn factor(&mut self) -> Result<Expr, String> {
        match self.peek() {
            Some(Token::Op('-')) => {
                self.next();
                Ok(Expr::Neg(Box::new(self.factor()?)))
            }
            Some(Token::Op('+')) => {
                self.next();
                self.factor()
            }
            _ => self.power(),
        }
    }

    fn power(&mut self) -> Result<Expr, String> {
        let base = self.atom()?;
        if let Some(Token::Op('^')) = self.peek() {
            self.next();
            // Right-associative: `2^3^2 == 2^(3^2)`, and the exponent may
            // carry a sign: `2^-1`.
            let exp = self.factor()?;
            return Ok(Expr::Binary(BinOp::Pow, Box::new(base), Box::new(exp)));
        }
        Ok(base)
    }

    fn atom(&mut self) -> Result<Expr, String> {
        match self.next() {
            Some(Token::Num(n)) => Ok(Expr::Const(n)),
            Some(Token::Ident(name)) => {
                // Function call?
                if let Some(Token::LParen) = self.peek() {
                    self.next();
                    let func = Func::from_name(&name)
                        .ok_or_else(|| format!("unknown function '{name}'"))?;
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Some(Token::RParen)) {
                        loop {
                            args.push(self.expr()?);
                            match self.next() {
                                Some(Token::Comma) => continue,
                                Some(Token::RParen) => break,
                                other => {
                                    return Err(format!(
                                        "expected ',' or ')' in {name}(), found {other:?}"
                                    ))
                                }
                            }
                        }
                    } else {
                        self.next(); // consume ')'
                    }
                    if args.len() != func.arity() {
                        return Err(format!(
                            "'{name}' expects {} argument(s), got {}",
                            func.arity(),
                            args.len()
                        ));
                    }
                    return Ok(Expr::Call(func, args));
                }
                // Variable or constant.
                Ok(match name.as_str() {
                    "x" | "X" => Expr::Var(Var::X),
                    "y" | "Y" => Expr::Var(Var::Y),
                    "pi" | "PI" => Expr::Const(std::f64::consts::PI),
                    "tau" | "TAU" => Expr::Const(std::f64::consts::TAU),
                    "e" | "E" => Expr::Const(std::f64::consts::E),
                    other => return Err(format!("unknown name '{other}'")),
                })
            }
            Some(Token::LParen) => {
                let e = self.expr()?;
                match self.next() {
                    Some(Token::RParen) => Ok(e),
                    other => Err(format!("expected ')', found {other:?}")),
                }
            }
            other => Err(format!("expected a number, name, or '(', found {other:?}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(src: &str, x: f64, y: f64) -> f64 {
        Expr::parse(src).unwrap().eval(x, y)
    }

    #[test]
    fn default_ripple_at_origin() {
        assert!((eval("cos(sqrt(x^2+y^2)*3)*0.4", 0.0, 0.0) - 0.4).abs() < 1e-12);
    }

    #[test]
    fn variables_and_functions() {
        assert!((eval("sin(x)*cos(y)", std::f64::consts::FRAC_PI_2, 0.0) - 1.0).abs() < 1e-12);
        assert_eq!(eval("min(x, y)", 3.0, 2.0), 2.0);
        assert_eq!(eval("atan2(y, x)", 1.0, 1.0), std::f64::consts::FRAC_PI_4);
    }

    #[test]
    fn precedence() {
        assert_eq!(eval("2+3*4", 0.0, 0.0), 14.0);
        assert_eq!(eval("(2+3)*4", 0.0, 0.0), 20.0);
        assert_eq!(eval("2^3*4", 0.0, 0.0), 32.0);
    }

    #[test]
    fn power_right_assoc_and_signed_exponent() {
        assert_eq!(eval("2^3^2", 0.0, 0.0), 512.0);
        assert_eq!(eval("2^-1", 0.0, 0.0), 0.5);
        // Unary minus binds looser than `^`: -(2^2)
        assert_eq!(eval("-2^2", 0.0, 0.0), -4.0);
    }

    #[test]
    fn constants_and_scientific_numbers() {
        assert!((eval("pi", 0.0, 0.0) - std::f64::consts::PI).abs() < 1e-12);
        assert_eq!(eval("1e3", 0.0, 0.0), 1000.0);
        assert_eq!(eval("2*e", 0.0, 0.0), 2.0 * std::f64::consts::E);
    }

    #[test]
    fn errors() {
        assert!(Expr::parse("x +").is_err());
        assert!(Expr::parse("foo(1)").is_err());
        assert!(Expr::parse("sin()").is_err());
        assert!(Expr::parse("sin(1, 2)").is_err());
        assert!(Expr::parse("(x").is_err());
        assert!(Expr::parse("x y").is_err());
        assert!(Expr::parse("1 /").is_err());
    }
}
