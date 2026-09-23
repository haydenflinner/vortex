// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native comparison of UTF-8 and binary arrays over canonical [`VarBinViewArray`]s.
//!
//! Equality first compares the leading 8 bytes of each view (length plus 4-byte prefix), which
//! answers most lanes without touching the data buffers. Ordering compares the inline 4-byte
//! prefixes first and only dereferences the full value on a prefix tie. UTF-8 values compare by
//! their byte representation, which matches code-point order. A constant operand compares as a
//! view of the constant.

use vortex_buffer::BitBuffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::Constant;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;
use crate::arrays::varbinview::ResolvedViews;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::binary::compare::compare_validity;
use crate::scalar_fn::fns::binary::compare::ordering_predicate;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::validity::Validity;

enum BytesOperand {
    Array {
        values: VarBinViewArray,
        validity: Validity,
    },
    Constant {
        value: Vec<u8>,
        validity: Validity,
    },
}

impl BytesOperand {
    fn try_new(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        if let Some(constant) = array.as_opt::<Constant>() {
            let value = constant_bytes(constant.scalar())?;
            return Ok(Self::Constant {
                value,
                validity: if constant.scalar().dtype().is_nullable() {
                    Validity::AllValid
                } else {
                    Validity::NonNullable
                },
            });
        }

        let values = array.clone().execute::<VarBinViewArray>(ctx)?;
        let validity = values.validity()?;
        Ok(Self::Array { values, validity })
    }

    fn validity(&self) -> Validity {
        match self {
            Self::Array { validity, .. } | Self::Constant { validity, .. } => validity.clone(),
        }
    }
}

fn constant_bytes(scalar: &Scalar) -> VortexResult<Vec<u8>> {
    let value = match scalar.dtype() {
        DType::Utf8(_) => scalar
            .as_utf8()
            .value()
            .map(|s| s.as_str().as_bytes().to_vec()),
        DType::Binary(_) => scalar.as_binary().value().map(|b| b.to_vec()),
        _ => vortex_bail!("expected utf8 or binary scalar, got {}", scalar.dtype()),
    };
    value.ok_or_else(|| vortex_err!("null constant handled by execute_compare"))
}

/// Compare two UTF-8 or binary arrays.
pub(super) fn compare_bytes(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = lhs.len();
    let lhs = BytesOperand::try_new(lhs, ctx)?;
    let rhs = BytesOperand::try_new(rhs, ctx)?;
    let validity = compare_validity(lhs.validity(), rhs.validity(), nullability)?;

    let bits = match (&lhs, &rhs) {
        (BytesOperand::Array { values: l, .. }, BytesOperand::Array { values: r, .. }) => {
            compare_views(
                &ResolvedViews::new(l),
                &ResolvedViews::new(r),
                op,
                ctx.allocator(),
            )
        }
        (BytesOperand::Array { values, .. }, BytesOperand::Constant { value, .. }) => {
            compare_views_constant(&ResolvedViews::new(values), value, op, ctx.allocator())
        }
        (BytesOperand::Constant { value, .. }, BytesOperand::Array { values, .. }) => {
            compare_views_constant(
                &ResolvedViews::new(values),
                value,
                op.swap(),
                ctx.allocator(),
            )
        }
        (BytesOperand::Constant { value: l, .. }, BytesOperand::Constant { value: r, .. }) => {
            // Unreachable through `execute_compare` (constant-constant is folded there), but
            // cheap to answer anyway.
            BitBuffer::full_in(
                ordering_predicate(op)(l.as_slice().cmp(r.as_slice())),
                len,
                ctx.allocator().clone(),
            )
        }
    };

    Ok(BoolArray::try_new(bits, validity)?.into_array())
}

fn compare_views(
    lhs: &ResolvedViews<'_>,
    rhs: &ResolvedViews<'_>,
    op: CompareOperator,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    let len = lhs.len();
    // The unchecked view accesses below index both sides with `i < len`, so this must hold even
    // in release builds.
    assert_eq!(len, rhs.len(), "compared views must have equal lengths");
    collect_compare_bits(
        len,
        // SAFETY: `collect_compare_bits` only asks for lanes below `len`, the length of both sides.
        |i| unsafe { lhs.view_unchecked(i) },
        lhs.buffers(),
        // SAFETY: as above.
        |i| unsafe { rhs.view_unchecked(i) },
        rhs.buffers(),
        op,
        allocator,
    )
}

fn compare_views_constant(
    lhs: &ResolvedViews<'_>,
    constant: &[u8],
    op: CompareOperator,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    // The constant as a view whose only buffer is the constant itself.
    let constant_view = BinaryView::make_view(constant, 0, 0);
    collect_compare_bits(
        lhs.len(),
        // SAFETY: `collect_compare_bits` only asks for lanes below `lhs.len()`.
        |i| unsafe { lhs.view_unchecked(i) },
        lhs.buffers(),
        |_| &constant_view,
        &[constant],
        op,
        allocator,
    )
}

/// Bit-pack `op` over `len` lanes, lane `i` comparing `lhs(i)` against `rhs(i)`.
///
/// The operator is dispatched outside the lane loop so each predicate inlines into its own loop;
/// a shared `fn(Ordering) -> bool` pointer would cost an indirect call per lane.
fn collect_compare_bits<'a>(
    len: usize,
    lhs: impl Fn(usize) -> &'a BinaryView,
    lhs_buffers: &[&[u8]],
    rhs: impl Fn(usize) -> &'a BinaryView,
    rhs_buffers: &[&[u8]],
    op: CompareOperator,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    let eq = |i| lhs(i).value_eq(lhs_buffers, rhs(i), rhs_buffers);
    let cmp = |i| lhs(i).value_cmp(lhs_buffers, rhs(i), rhs_buffers);
    let allocator = allocator.clone();
    match op {
        CompareOperator::Eq => BitBuffer::collect_bool_in(len, eq, allocator),
        CompareOperator::NotEq => BitBuffer::collect_bool_in(len, |i| !eq(i), allocator),
        CompareOperator::Gt => BitBuffer::collect_bool_in(len, |i| cmp(i).is_gt(), allocator),
        CompareOperator::Gte => BitBuffer::collect_bool_in(len, |i| cmp(i).is_ge(), allocator),
        CompareOperator::Lt => BitBuffer::collect_bool_in(len, |i| cmp(i).is_lt(), allocator),
        CompareOperator::Lte => BitBuffer::collect_bool_in(len, |i| cmp(i).is_le(), allocator),
    }
}
