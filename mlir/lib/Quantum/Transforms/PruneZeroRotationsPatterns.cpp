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

#define DEBUG_TYPE "ions-decomposition"

#include <variant>

#include "mlir/Dialect/Arith/IR/Arith.h"

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

struct PruneZeroRotationsRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        ValueRange inQubits = op.getInQubits();
        auto parent = inQubits[0].getDefiningOp();

        if (!isa<CustomOp>(parent)) {
            return failure();
        }
        
        auto parentOp = llvm::cast<CustomOp>(inQubits[0].getDefiningOp());

        std::vector<std::string> rotationGates = {"RX", "RY", "RZ"};

        auto parentOpGateIndex = std::find(rotationGates.begin(), rotationGates.end(), parentOp.getGateName().str());
        auto opGateIndex = std::find(rotationGates.begin(), rotationGates.end(), op.getGateName().str());
        
        if (parentOpGateIndex == rotationGates.end() || opGateIndex == rotationGates.end()) {
            return failure();
        }

        mlir::Value parentAngle = parentOp.getParams().front();
        arith::ConstantFloatOp parentAngleDefiningOp = parentAngle.getDefiningOp<arith::ConstantFloatOp>();
        std::optional<double> parentAngleOpt = getStaticValueOrNothing(parentAngleDefiningOp);
        bool parentAngleIsZero = parentAngleOpt.has_value() && parentAngleOpt.value() == 0.0;
        

        mlir::Value angle = op.getParams().front();
        arith::ConstantFloatOp angleDefiningOp = angle.getDefiningOp<arith::ConstantFloatOp>();
        std::optional<double> angleOpt = getStaticValueOrNothing(angleDefiningOp);
        bool angleIsZero = angleOpt.has_value() && angleOpt.value() == 0.0;
        
        CustomOp prunedOp;
        if (parentAngleIsZero) {
            prunedOp =
                rewriter.create<CustomOp>(op.getLoc(), op.getOutQubits().getTypes(), TypeRange{}, op.getParams().front(),
                                        parentOp.getInQubits(), op.getGateName().str(), false, ValueRange{}, ValueRange{});
        }
        else if (angleIsZero) {
            prunedOp =
                rewriter.create<CustomOp>(parentOp.getLoc(), op.getOutQubits().getTypes(), TypeRange{}, parentOp.getParams().front(),
                                        parentOp.getInQubits(), parentOp.getGateName().str(), false, ValueRange{}, ValueRange{});
        }
        else {
            return failure();
        }

        rewriter.replaceOp(op, prunedOp);
        rewriter.eraseOp(parentOp);
        return success();
    }
};
} // namespace

namespace catalyst {
namespace quantum {

void populatePruneZeroRotationsPatterns(RewritePatternSet &patterns)
{
    patterns.add<PruneZeroRotationsRewritePattern>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
