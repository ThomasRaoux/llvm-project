//===- LinalgTransforms.cpp - Linalg transformations as patterns ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic and helpers to expose Linalg transforms as rewrite
// patterns.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/EDSC/Intrinsics.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <type_traits>

#define DEBUG_TYPE "linalg-transforms"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

//===----------------------------------------------------------------------===//
// Transformations exposed as rewrite patterns.
//===----------------------------------------------------------------------===//
// Marker used as attribute name in generated Linalg rewriting transformations.
const StringLiteral mlir::linalg::LinalgTransforms::kLinalgTransformMarker =
    "__internal_linalg_transform__";

mlir::linalg::LinalgMarker::LinalgMarker(ArrayRef<Identifier> matchDisjunction,
                                         Optional<Identifier> replacement)
    : matchDisjunction(matchDisjunction.begin(), matchDisjunction.end()),
      replacement(replacement) {}

LogicalResult
mlir::linalg::LinalgMarker::checkAndNotify(PatternRewriter &rewriter,
                                           Operation *op) const {
  auto attr = op->template getAttrOfType<StringAttr>(
      LinalgTransforms::kLinalgTransformMarker);

  if (!attr) {
    // 1. Has no marker case and matchDisjunction is empty.
    if (matchDisjunction.empty())
      return success();

    // 2. Has no marker but was expecting a marker.
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << " does not have any marker from list: ";
      interleaveComma(matchDisjunction, diag);
    });
  }

  // 4. Match explicit marker.
  for (auto marker : matchDisjunction)
    if (attr.getValue() == marker)
      return success();

  // 5. Fail to match.
  return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
    diag << " does not have any marker from list: ";
    interleaveComma(matchDisjunction, diag);
  });
}

void mlir::linalg::LinalgMarker::replaceLinalgMarker(PatternRewriter &rewriter,
                                                     Operation *op) const {
  if (replacement.hasValue())
    op->setAttr(LinalgTransforms::kLinalgTransformMarker,
                rewriter.getStringAttr(replacement.getValue()));
  else
    op->removeAttr(Identifier::get(LinalgTransforms::kLinalgTransformMarker,
                                   rewriter.getContext()));
}

LinalgTilingOptions &
mlir::linalg::LinalgTilingOptions::setTileSizes(ArrayRef<int64_t> ts) {
  SmallVector<int64_t, 4> tileSizes(ts.begin(), ts.end());
  tileSizeComputationFunction = [tileSizes](OpBuilder &b, Operation *op) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(
        &op->getParentOfType<FuncOp>().getBody().front());
    return llvm::to_vector<4>(map_range(tileSizes, [&](int64_t s) {
      Value v = b.create<ConstantIndexOp>(op->getLoc(), s);
      return v;
    }));
  };
  return *this;
}

/// Linalg base tiling pattern.
mlir::linalg::LinalgBaseTilingPattern::LinalgBaseTilingPattern(
    StringRef opName, MLIRContext *context, LinalgTilingOptions options,
    LinalgMarker marker, PatternBenefit benefit)
    : RewritePattern(opName, {}, benefit, context), marker(marker),
      options(options) {}

LogicalResult mlir::linalg::LinalgBaseTilingPattern::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  LinalgOp linalgOp = dyn_cast<LinalgOp>(op);
  if (!linalgOp)
    return failure();
  if (failed(marker.checkAndNotify(rewriter, linalgOp)))
    return failure();

  Optional<TiledLinalgOp> res = tileLinalgOp(rewriter, linalgOp, options);

  if (!res)
    return failure();

  // New marker if specified.
  marker.replaceLinalgMarker(rewriter, res->op.getOperation());

  rewriter.eraseOp(op);
  return success();
}

/// Linalg base interchange pattern.
mlir::linalg::LinalgBaseInterchangePattern::LinalgBaseInterchangePattern(
    StringRef opName, MLIRContext *context,
    ArrayRef<unsigned> interchangeVector, LinalgMarker marker,
    PatternBenefit benefit)
    : RewritePattern(opName, {}, benefit, context), marker(marker),
      interchangeVector(interchangeVector.begin(), interchangeVector.end()) {}

LogicalResult mlir::linalg::LinalgBaseInterchangePattern::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  LinalgOp linalgOp = dyn_cast<LinalgOp>(op);
  if (!linalgOp)
    return failure();
  if (failed(marker.checkAndNotify(rewriter, linalgOp)))
    return failure();
  if (failed(interchangeGenericLinalgOpPrecondition(op, interchangeVector)))
    return failure();

  // TODO: figure out how this interplays with named ops. In particular this
  // should break the named op property.
  rewriter.updateRootInPlace(op, [&]() {
    interchange(linalgOp, interchangeVector);
    // New marker if specified.
    marker.replaceLinalgMarker(rewriter, op);
  });
  return success();
}

mlir::linalg::LinalgBasePromotionPattern::LinalgBasePromotionPattern(
    StringRef opName, MLIRContext *context, LinalgPromotionOptions options,
    LinalgMarker marker, PatternBenefit benefit)
    : RewritePattern(opName, {}, benefit, context), marker(marker),
      options(options) {}

LogicalResult mlir::linalg::LinalgBasePromotionPattern::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  if (failed(marker.checkAndNotify(rewriter, op)))
    return failure();
  if (failed(promoteSubviewsPrecondition(op, options)))
    return failure();

  // TODO: We cannot use root update here. This pattern is creating other ops,
  // so if the promotion fails, those need to be cleaned up, which doesnt seem
  // to be happening here. So to fail properly, we should be cloning the op and
  // deleting the previous op. This needs more investigation.
  rewriter.startRootUpdate(op);
  Optional<LinalgOp> promotedOp = promoteSubViews(rewriter, op, options);
  if (!promotedOp) {
    rewriter.cancelRootUpdate(op);
    return op->emitError("subview promotion failed");
  }
  rewriter.finalizeRootUpdate(op);
  marker.replaceLinalgMarker(rewriter, op);
  return success();
}

mlir::linalg::LinalgBaseVectorizationPattern::LinalgBaseVectorizationPattern(
    StringRef opName, MLIRContext *context, LinalgMarker marker,
    PatternBenefit benefit)
    : RewritePattern(opName, {}, benefit, context), marker(marker) {}

LogicalResult mlir::linalg::LinalgBaseVectorizationPattern::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  LinalgOp linalgOp = dyn_cast<LinalgOp>(op);
  if (!linalgOp)
    return failure();
  if (failed(marker.checkAndNotify(rewriter, linalgOp)))
    return failure();
  if (failed(vectorizeLinalgOpPrecondition(op)))
    return failure();
  vectorizeLinalgOp(rewriter, op);
  rewriter.eraseOp(op);
  return success();
}

LogicalResult mlir::linalg::applyStagedPatterns(
    Operation *op, ArrayRef<OwningRewritePatternList> stage1Patterns,
    const OwningRewritePatternList &stage2Patterns,
    function_ref<LogicalResult(Operation *)> stage3Lambda) {
  unsigned iteration = 0;
  (void)iteration;
  for (const auto &patterns : stage1Patterns) {
    LLVM_DEBUG(DBGS() << "Before 1st stage, iter: " << ++iteration << "\n"
                      << *op);
    if (failed(applyPatternsAndFoldGreedily(op, patterns))) {
      LLVM_DEBUG(DBGS() << "Underlying first stage rewrite did not converge");
      return failure();
    }
    LLVM_DEBUG(DBGS() << "After 1st stage, iter: " << ++iteration << "\n"
                      << *op);
    if (failed(applyPatternsAndFoldGreedily(op, stage2Patterns))) {
      LLVM_DEBUG(DBGS() << "Underlying 2nd stage rewrite did not converge");
      return failure();
    }
    LLVM_DEBUG(DBGS() << "After 2nd stage, iter : " << iteration << "\n"
                      << *op);
    if (stage3Lambda) {
      if (failed(stage3Lambda(op)))
        return failure();
      LLVM_DEBUG(DBGS() << "After 3rd stage, iter : " << iteration << "\n"
                        << *op);
    }
  }
  return success();
}

/// Substitute the AffineExprDim at position `dimIdx`, which corresponds to a
/// loop induction variable (e.g. scf.for %iv = %lb to %ub step %step) by the
/// AffineExpr representing `%lb + %step * floorDiv(%iv - %lb, %step)` such
/// that:
/// 1. the AffineExpr for %lb is either an AffineConstantExpr or an
///    AffineDimExpr depending on whether the value is constant or not.
/// 2. the AffineExpr for %step is either an AffineConstantExpr or an
///    AffineSymbolExpr depending on whether the value is constant or not.
static void substituteLoop(unsigned dimIdx, Value lbVal, Value ubVal,
                           Value stepVal, SmallVectorImpl<AffineExpr> &exprs,
                           SmallVectorImpl<Value> &dims,
                           SmallVectorImpl<Value> &symbols) {
  MLIRContext *ctx = lbVal.getContext();

  // 1. maybe add a new dim for the `lb`.
  auto lbConstant = lbVal.getDefiningOp<ConstantIndexOp>();
  AffineExpr lb = lbConstant ? getAffineConstantExpr(lbConstant.getValue(), ctx)
                             : getAffineDimExpr(dims.size(), ctx);
  if (!lbConstant)
    dims.push_back(lbVal);

  // 2. maybe add a new symbol for the `step`.
  auto stepConstant = stepVal.getDefiningOp<ConstantIndexOp>();
  AffineExpr step = stepConstant
                        ? getAffineConstantExpr(stepConstant.getValue(), ctx)
                        : getAffineSymbolExpr(symbols.size(), ctx);
  if (!stepConstant)
    symbols.push_back(stepVal);

  // 3. Rewrite `exprs` in place by replacing `dim[dimIdx]` by `lb + step * iv`.
  AffineExpr iv = getAffineDimExpr(dimIdx, ctx);
  for (auto &e : exprs)
    e = e.replace(iv, lb + step * (iv - lb).floorDiv(step));
}

/// Traverse the `dims` and substitute linear expressions in place of induction
/// variables in `exprs`.
static void substitute(SmallVectorImpl<AffineExpr> &exprs,
                       SmallVectorImpl<Value> &dims,
                       SmallVectorImpl<Value> &symbols) {
  assert(!exprs.empty() && "Unexpected empty exprs");
  LLVM_DEBUG(llvm::interleaveComma(dims, DBGS() << "Start subst with dims: "));
  LLVM_DEBUG(llvm::dbgs() << "\n");

  // Note: `dims` and `symbols` grow as we iterate, upper bound is dynamic.
  for (unsigned dimIdx = 0; dimIdx < dims.size(); ++dimIdx) {
    Value dim = dims[dimIdx];
    LLVM_DEBUG(DBGS() << "Subst: " << dim << "\n");

    // Replace dim @ pos[dimIdx] by `%lb + %step * new_dim`
    // Where new dim / symbols are added depending on whether the values are
    // static or not.
    if (auto forOp = scf::getForInductionVarOwner(dim)) {
      substituteLoop(dimIdx, forOp.lowerBound(), forOp.upperBound(),
                     forOp.step(), exprs, dims, symbols);
      continue;
    }
    if (auto parallelForOp = scf::getParallelForInductionVarOwner(dim)) {
      for (unsigned idx = 0, e = parallelForOp.getNumLoops(); idx < e; ++idx)
        substituteLoop(dimIdx, parallelForOp.lowerBound()[idx],
                       parallelForOp.upperBound()[idx],
                       parallelForOp.step()[idx], exprs, dims, symbols);
      continue;
    }
  }

  // Cleanup and simplify the results.
  SmallVector<Value, 4> operands(dims.begin(), dims.end());
  operands.append(symbols.begin(), symbols.end());
  auto map = AffineMap::get(dims.size(), symbols.size(), exprs,
                            exprs.front().getContext());
  // Pull in affine.apply operations and compose them fully into the result.
  fullyComposeAffineMapAndOperands(&map, &operands);
  canonicalizeMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  // Assign the results.
  exprs.assign(map.getResults().begin(), map.getResults().end());
  dims.assign(operands.begin(), operands.begin() + map.getNumDims());
  symbols.assign(operands.begin() + map.getNumDims(), operands.end());
}

LogicalResult AffineMinSCFCanonicalizationPattern::matchAndRewrite(
    AffineMinOp minOp, PatternRewriter &rewriter) const {
  // At least one loop is needed to canonicalize affine.min + SCF.
  auto isLoopLike = [](Value v) {
    return scf::getParallelForInductionVarOwner(v) ||
           scf::getForInductionVarOwner(v);
  };
  if (llvm::none_of(minOp.getDimOperands(), isLoopLike))
    return failure();

  LLVM_DEBUG(DBGS() << "Canonicalize AffineMinSCF: " << *minOp.getOperation()
                    << "\n");

  auto exprs = llvm::to_vector<4>(minOp.getAffineMap().getResults());
  SmallVector<Value, 4> dims(minOp.getDimOperands()),
      symbols(minOp.getSymbolOperands());
  substitute(exprs, dims, symbols);

  MLIRContext *ctx = minOp.getContext();
  auto map = AffineMap::get(dims.size(), symbols.size(), exprs, ctx);
  LLVM_DEBUG(DBGS() << "Resulting map: " << map << "\n");

  // Check whether any of the expressions divides all expressions. In which case
  // it is guaranteed to be the min.
  for (auto e : map.getResults()) {
    LLVM_DEBUG(DBGS() << "Candidate mod: " << e << "\n");
    if (!e.isSymbolicOrConstant())
      continue;

    LLVM_DEBUG(DBGS() << "Check whether mod: " << e << " is zero\n");
    SmallVector<AffineExpr, 4> modExprs;
    for (auto ee : map.getResults())
      modExprs.push_back(ee % e);

    AffineMap modMap = simplifyAffineMap(
        AffineMap::get(map.getNumDims(), map.getNumSymbols(), modExprs, ctx));
    LLVM_DEBUG(DBGS() << "simplified modMap: " << modMap << "\n");

    auto isZero = [](AffineExpr e) {
      if (auto cst = e.dyn_cast<AffineConstantExpr>())
        return cst.getValue() == 0;
      return false;
    };
    if (llvm::all_of(modMap.getResults(), isZero)) {
      if (auto cst = e.dyn_cast<AffineConstantExpr>()) {
        rewriter.replaceOpWithNewOp<ConstantIndexOp>(minOp, cst.getValue());
      } else {
        auto resultMap =
            AffineMap::get(map.getNumDims(), map.getNumSymbols(), {e}, ctx);
        SmallVector<Value, 4> resultOperands = dims;
        resultOperands.append(symbols.begin(), symbols.end());
        canonicalizeMapAndOperands(&resultMap, &resultOperands);
        resultMap = simplifyAffineMap(resultMap);
        rewriter.replaceOpWithNewOp<AffineApplyOp>(minOp, resultMap,
                                                   resultOperands);
      }
      return success();
    }
  }

  return failure();
}
