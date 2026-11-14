//! LAPACK triangular, packed and RFP block coordinates.
//! RFP follows the format documented by Reference-LAPACK SRC/*trttf.f.
#[derive(Clone, Copy)]
pub(crate) enum TriangleStorage {
    Full(isize),
    Packed,
    Rfp { transposed: bool },
}
pub(crate) fn rfp_location(
    n: usize,
    lower: bool,
    transposed: bool,
    i: usize,
    j: usize,
) -> (usize, isize, bool) {
    let even = n % 2 == 0;
    let rows = if even { n + 1 } else { n };
    let columns = (n + 1) / 2;
    let half = n / 2;
    let (row, column, folded) = if lower {
        if j < n - half {
            (i + usize::from(even), j, false)
        } else {
            (j - (n - half), i - half, true)
        }
    } else if j >= half {
        (i, j - half, false)
    } else {
        (half + j + 1, i, true)
    };
    let (index, stride) = if transposed {
        (column + row * columns, if folded { 1 } else { columns })
    } else {
        (row + column * rows, if folded { rows } else { 1 })
    };
    (index, stride as isize, folded ^ transposed)
}
impl TriangleStorage {
    pub(crate) fn segment(self, n: usize, lower: bool, j: usize) -> (usize, isize, bool) {
        let i = if lower { j } else { 0 };
        match self {
            Self::Full(ld) => (i + j * ld as usize, 1, false),
            Self::Packed => (
                if lower {
                    j * n - j * (j + 1) / 2 + i
                } else {
                    j * (j + 1) / 2 + i
                },
                1,
                false,
            ),
            Self::Rfp { transposed } => rfp_location(n, lower, transposed, i, j),
        }
    }
}
