// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "prune-zero-rotations"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::quantum;

std::optional<double> getStaticValueOrNothing(const Value value)
{

    std::optional<double> staticValue;
    if (auto constOp = value.getDefiningOp();
        constOp && constOp->hasTrait<OpTrait::ConstantLike>()) {
        if (auto floatAttr = constOp->getAttrOfType<FloatAttr>("value")) {
            staticValue = floatAttr.getValueAsDouble();
        }
    }
    return staticValue;
}

namespace {

struct PruneAfterZeroRotationsRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        std::vector<std::string> rotationGates = {"RX", "RY", "RZ"};

        ValueRange inQubits = op.getInQubits();
        auto opGateIndex =
            std::find(rotationGates.begin(), rotationGates.end(), op.getGateName().str());

        if (opGateIndex == rotationGates.end()) {
            return failure();
        }

        mlir::Value angle = op.getParams().front();
        arith::ConstantFloatOp angleDefiningOp = angle.getDefiningOp<arith::ConstantFloatOp>();
        std::optional<double> angleOpt = getStaticValueOrNothing(angleDefiningOp);
        bool angleIsZero = angleOpt.has_value() && angleOpt.value() == 0.0;

        if (!angleIsZero) {
            return failure();
        }

        mlir::Value outQubit = op.getOutQubits()[0];
        outQubit.replaceAllUsesWith(inQubits[0]);
        rewriter.eraseOp(op);

        return success();
    }
};

struct PruneBeforeZeroRotationRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        std::vector<std::string> rotationGates = {"RX", "RY", "RZ"};

        ValueRange inQubits = op.getInQubits();

        std::vector<CustomOp> prunedParentOps;
        for (Value inQubit : inQubits) {
            if (!isa<CustomOp>(inQubit.getDefiningOp())) {
                continue;
            }

            CustomOp parentOp = inQubit.getDefiningOp<CustomOp>();
            auto parentOpGateIndex =
                std::find(rotationGates.begin(), rotationGates.end(), parentOp.getGateName().str());

            if (parentOpGateIndex == rotationGates.end()) {
                continue;
            }

            mlir::Value parentAngle = parentOp.getParams().front();
            arith::ConstantFloatOp parentAngleDefiningOp =
                parentAngle.getDefiningOp<arith::ConstantFloatOp>();
            std::optional<double> parentAngleOpt = getStaticValueOrNothing(parentAngleDefiningOp);
            bool parentAngleIsZero = parentAngleOpt.has_value() && parentAngleOpt.value() == 0.0;

            if (!parentAngleIsZero) {
                continue;
            }

            inQubit.replaceAllUsesWith(parentOp.getInQubits()[0]);
            prunedParentOps.push_back(parentOp);
        }

        for (CustomOp parent : prunedParentOps) {
            rewriter.eraseOp(parent);
        }

        return failure();
    }
};
} // namespace

namespace catalyst {
namespace quantum {

void populatePruneZeroRotationsPatterns(RewritePatternSet &patterns)
{
    patterns.add<PruneBeforeZeroRotationRewritePattern>(patterns.getContext(), 1);
    patterns.add<PruneAfterZeroRotationsRewritePattern>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
