//! Native distribution sampling and an independently owned StdRng stream.
use crate::collections::{VectorData, VectorOwner, VectorStore};
use crate::typed::{MatrixData, TypedMatrix};
use crate::{guarded, OwnedMatrix};
use faer::stats::{
    ComplexDistribution, CwiseColDistribution, CwiseMatDistribution, CwiseRowDistribution,
    UnitaryMat,
};
use faer::traits::ComplexField;
use faer::{Col, Mat, Row};
use rand::{
    distr::{Distribution, StandardUniform},
    rngs::StdRng,
    RngCore, SeedableRng,
};
use rand_distr::{Normal, StandardNormal};
use rustcall_julia_macros::julia;
#[julia]
#[derive(Clone)]
pub struct OwnedRandom {
    rng: StdRng,
}
#[julia]
impl OwnedRandom {
    #[julia]
    pub fn new() -> Self {
        Self {
            rng: StdRng::seed_from_u64(0),
        }
    }
    #[julia]
    pub fn rng_seed(&mut self, seed: u64) -> Result<(), String> {
        guarded(|| {
            self.rng = StdRng::seed_from_u64(seed);
            Ok(())
        })
    }
    #[julia]
    pub fn rng_next(&mut self) -> Result<u64, String> {
        guarded(|| Ok(self.rng.next_u64()))
    }
    #[julia]
    pub fn duplicate(&self) -> Result<Self, String> {
        guarded(|| Ok(self.clone()))
    }
}
enum Generated<T> {
    Matrix(Mat<T>),
    Column(Col<T>),
    Row(Row<T>),
}
fn draw<T: ComplexField, D: Distribution<T>>(
    rng: &mut StdRng,
    m: usize,
    n: usize,
    kind: i32,
    unitary: bool,
    dist: D,
) -> Result<Generated<T>, String> {
    if unitary {
        if kind != 0 || m != n {
            return Err("unitary samples must be square matrices".into());
        }
        return Ok(Generated::Matrix(if m == 0 {
            Mat::zeros(0, 0)
        } else {
            UnitaryMat {
                dim: m,
                standard_normal: &dist,
            }
            .sample(rng)
        }));
    }
    Ok(match kind {
        0 => Generated::Matrix(
            CwiseMatDistribution {
                nrows: m,
                ncols: n,
                dist: &dist,
            }
            .sample(rng),
        ),
        1 if n == 1 => Generated::Column(
            CwiseColDistribution {
                nrows: m,
                dist: &dist,
            }
            .sample(rng),
        ),
        2 if m == 1 => Generated::Row(
            CwiseRowDistribution {
                ncols: n,
                dist: &dist,
            }
            .sample(rng),
        ),
        _ => return Err("random collection dimensions/kind mismatch".into()),
    })
}
macro_rules! real_generator {
    ($name:ident,$t:ty) => {
        fn $name(
            rng: &mut StdRng,
            m: usize,
            n: usize,
            kind: i32,
            normal: bool,
            unitary: bool,
        ) -> Result<Generated<$t>, String> {
            if normal || unitary {
                draw(rng, m, n, kind, unitary, StandardNormal)
            } else {
                draw(rng, m, n, kind, false, StandardUniform)
            }
        }
    };
}
macro_rules! complex_generator {
    ($name:ident,$t:ty,$r:ty,$sigma:expr) => {
        fn $name(
            rng: &mut StdRng,
            m: usize,
            n: usize,
            kind: i32,
            normal: bool,
            unitary: bool,
        ) -> Result<Generated<$t>, String> {
            if normal || unitary {
                let gaussian = Normal::<$r>::new(0.0, $sigma).map_err(|e| format!("{e:?}"))?;
                draw(
                    rng,
                    m,
                    n,
                    kind,
                    unitary,
                    ComplexDistribution::new(gaussian, gaussian),
                )
            } else {
                draw(
                    rng,
                    m,
                    n,
                    kind,
                    false,
                    ComplexDistribution::new(StandardUniform, StandardUniform),
                )
            }
        }
    };
}
real_generator!(f32_sample, f32);
real_generator!(f64_sample, f64);
complex_generator!(c32_sample, faer::c32, f32, std::f32::consts::FRAC_1_SQRT_2);
complex_generator!(c64_sample, faer::c64, f64, std::f64::consts::FRAC_1_SQRT_2);
fn matrix<T>(generated: Generated<T>) -> Result<Mat<T>, String> {
    match generated {
        Generated::Matrix(a) => Ok(a),
        _ => Err("matrix sampling requires matrix layout".into()),
    }
}
fn vector<T>(generated: Generated<T>) -> Result<VectorStore<T>, String> {
    match generated {
        Generated::Column(a) => Ok(VectorStore::Column(a)),
        Generated::Row(a) => Ok(VectorStore::Row(a)),
        _ => Err("collection sampling requires column or row layout".into()),
    }
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn random_from(
        &mut self,
        rng: &mut OwnedRandom,
        m: usize,
        n: usize,
        normal: bool,
        unitary: bool,
    ) -> Result<(), String> {
        guarded(|| {
            self.matrix = matrix(f64_sample(&mut rng.rng, m, n, 0, normal, unitary)?)?;
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn random_from(
        &mut self,
        rng: &mut OwnedRandom,
        m: usize,
        n: usize,
        normal: bool,
        unitary: bool,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| {
            self.data = Some(match tag {
                0 => MatrixData::F32(matrix(f32_sample(&mut rng.rng, m, n, 0, normal, unitary)?)?),
                1 => MatrixData::F64(matrix(f64_sample(&mut rng.rng, m, n, 0, normal, unitary)?)?),
                2 => MatrixData::C32(matrix(c32_sample(&mut rng.rng, m, n, 0, normal, unitary)?)?),
                3 => MatrixData::C64(matrix(c64_sample(&mut rng.rng, m, n, 0, normal, unitary)?)?),
                _ => return Err("unsupported random scalar".into()),
            });
            Ok(())
        })
    }
}
#[julia]
impl VectorOwner {
    #[julia]
    pub fn random_from(
        &mut self,
        rng: &mut OwnedRandom,
        m: usize,
        n: usize,
        normal: bool,
        kind: i32,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| {
            self.data = Some(match tag {
                0 => VectorData::F32(vector(f32_sample(
                    &mut rng.rng,
                    m,
                    n,
                    kind,
                    normal,
                    false,
                )?)?),
                1 => VectorData::F64(vector(f64_sample(
                    &mut rng.rng,
                    m,
                    n,
                    kind,
                    normal,
                    false,
                )?)?),
                2 => VectorData::C32(vector(c32_sample(
                    &mut rng.rng,
                    m,
                    n,
                    kind,
                    normal,
                    false,
                )?)?),
                3 => VectorData::C64(vector(c64_sample(
                    &mut rng.rng,
                    m,
                    n,
                    kind,
                    normal,
                    false,
                )?)?),
                _ => return Err("unsupported random scalar".into()),
            });
            Ok(())
        })
    }
}
