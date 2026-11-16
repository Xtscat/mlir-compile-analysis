//===- iree-harness.cpp - run an IREE pass pipeline with the collector ----===//
//
// Driver that links the IREE compiler and runs one of IREE's own pipelines in
// process with include/CompileAnalysis.h installed. Unlike ../iree/harness.cpp
// it does not bring its own dialects and patterns: the point here is to measure
// the real pipelines (flow, stream, hal, executable sources) and the real
// rewrite drivers inside them.
//
// The plugin and target-registry setup mirrors iree-opt, which lives in
// compiler/src/iree/compiler/API/Internal/IREEOptToolEntryPoint.cpp, because a
// pipeline that materializes executables needs the registered HAL target
// devices and backends to resolve them.
//
//   iree-harness <input.mlir> [--pipeline=<text>] [--demo-patterns]
//
// --demo-patterns additionally runs a two-pattern greedy rewrite on a small
// module of its own with the collector's listener attached. That is the only
// way to check the outcome and match-failure-reason fields from inside this
// driver: IREE's own passes build their GreedyRewriteConfig internally and give
// a listener nowhere to attach.
//
//===----------------------------------------------------------------------===//

#include "CompileAnalysis.h"

#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/PluginAPI/PluginManager.h"
#include "iree/compiler/Tools/init_dialects.h"
#include "iree/compiler/Tools/init_passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::iree_compiler;

static llvm::cl::opt<std::string> inputFile(llvm::cl::Positional,
                                            llvm::cl::desc("<input file>"),
                                            llvm::cl::init("-"));

static llvm::cl::opt<std::string> pipeline(
    "pipeline",
    llvm::cl::desc("IREE pass pipeline to run, without the builtin.module "
                   "wrapper"),
    llvm::cl::init("iree-flow-transformation-pipeline"));

static llvm::cl::opt<bool> noVerify(
    "no-verify",
    llvm::cl::desc("skip the pass-manager verifier, to see how much of a "
                   "pass's `outside` time it accounts for"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> demoPatterns(
    "demo-patterns",
    llvm::cl::desc("also run a local two-pattern greedy rewrite with the "
                   "collector's pattern listener attached"),
    llvm::cl::init(false));

namespace {

/// Rewrites a zero integer constant into a different constant. The fold hook
/// never rewrites a constant into another constant, so this attempt really
/// reaches a pattern and the success outcome is recorded.
struct ZeroToSevenPattern : mlir::RewritePattern {
  explicit ZeroToSevenPattern(mlir::MLIRContext *context)
      : RewritePattern(mlir::arith::ConstantOp::getOperationName(),
                       /*benefit=*/2, context) {
    setDebugName("ZeroToSeven");
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto constant = llvm::cast<mlir::arith::ConstantOp>(op);
    auto value = llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue());
    if (!value)
      return rewriter.notifyMatchFailure(op, "constant is not an integer");
    if (value.getInt() != 0)
      return rewriter.notifyMatchFailure(op, "constant is not zero");
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(
        op, rewriter.getIntegerAttr(value.getType(), 7));
    return mlir::success();
  }
};

/// Always fails, which pins the failure outcome and its reason down on a known
/// operation kind.
struct NeverFoldsConstantPattern : mlir::RewritePattern {
  explicit NeverFoldsConstantPattern(mlir::MLIRContext *context)
      : RewritePattern(mlir::arith::ConstantOp::getOperationName(),
                       /*benefit=*/1, context) {
    setDebugName("NeverFoldsConstant");
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    return rewriter.notifyMatchFailure(op, "demo: constants are left alone");
  }
};

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);

  // Same registration iree-opt performs: every MLIR and IREE dialect plus the
  // in-tree compiler modules, and every MLIR and IREE pass.
  mlir::DialectRegistry registry;
  registerAllDialects(registry);

  // Plugins in the form iree-opt supports them: global initialization, dialect
  // registration and command-line options, all before parsing the options.
  PluginManager pluginManager;
  if (!pluginManager.loadAvailablePlugins()) {
    llvm::errs() << "iree-harness: cannot load plugins\n";
    return 1;
  }
  pluginManager.globalInitialize();
  pluginManager.registerPasses();
  pluginManager.registerGlobalDialects(registry);
  pluginManager.initializeCLI();
  registerAllPasses();

  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "iree pipeline harness\n");

  auto &pluginManagerOptions = PluginManagerOptions::FromFlags::get();
  auto localBinder = OptionsBinder::local();
  PluginManagerSession pluginSession(pluginManager, localBinder,
                                     pluginManagerOptions);
  if (failed(pluginSession.initializePlugins())) {
    llvm::errs() << "iree-harness: cannot initialize plugins\n";
    return 1;
  }
  pluginSession.registerDialects(registry);

  // Outside the compiler driver there is no per-session registry, so the plugin
  // devices and backends go into the global registry, exactly as iree-opt does.
  IREE::HAL::TargetDeviceList targetDeviceList;
  pluginSession.populateHALTargetDevices(targetDeviceList);
  const_cast<IREE::HAL::TargetRegistry &>(IREE::HAL::TargetRegistry::getGlobal())
      .mergeFrom(targetDeviceList);
  IREE::HAL::TargetBackendList targetBackendList;
  pluginSession.populateHALTargetBackends(targetBackendList);
  const_cast<IREE::HAL::TargetRegistry &>(IREE::HAL::TargetRegistry::getGlobal())
      .mergeFrom(targetBackendList);

  MLIRContext context(registry);
  context.allowUnregisteredDialects(false);

  // The collector is off unless COMPILE_ANALYSIS names an output file.
  compile_analysis::installIfRequested(context);

  llvm::SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler diagnostics(sourceMgr, &context);
  auto module = parseSourceFile<ModuleOp>(inputFile, &context);
  if (!module) {
    llvm::errs() << "iree-harness: cannot parse '" << inputFile << "'\n";
    return 1;
  }

  // One pass manager per stage, run in the order given. iree-compile does the
  // same thing phase by phase, and splitting them here lets each stage be
  // marked in the trace: the action stream alone does not say where one
  // pipeline ends and the next begins.
  //
  // Each manager is anchored on the module itself: `builtin.module(...)`
  // parses into an adaptor that looks for a nested builtin.module and silently
  // runs nothing.
  llvm::SmallVector<llvm::StringRef> stages;
  llvm::StringRef(pipeline).split(stages, ',', /*MaxSplit=*/-1,
                                 /*KeepEmpty=*/false);
  if (stages.empty()) {
    llvm::errs() << "iree-harness: empty pipeline '\n" << pipeline << "'\n";
    return 1;
  }
  for (llvm::StringRef stage : stages) {
    PassManager passManager(&context, ModuleOp::getOperationName());
    if (failed(parsePassPipeline(stage, passManager))) {
      llvm::errs() << "iree-harness: cannot parse pipeline '" << stage << "'\n";
      return 1;
    }
    passManager.enableVerifier(!noVerify);
    compile_analysis::markStage(stage);
    if (failed(passManager.run(*module))) {
      llvm::errs() << "iree-harness: pipeline failed in stage '" << stage
                   << "'\n";
      return 1;
    }
  }
  module->print(llvm::outs());
  llvm::outs() << "\n";

  if (demoPatterns) {
    auto demo = parseSourceString<ModuleOp>(R"mlir(
func.func @demo() -> i32 {
  %zero = arith.constant 0 : i32
  return %zero : i32
}
)mlir", &context);
    if (!demo) {
      llvm::errs() << "iree-harness: cannot parse the demo module\n";
      return 1;
    }
    mlir::RewritePatternSet patterns(&context);
    patterns.add<ZeroToSevenPattern, NeverFoldsConstantPattern>(&context);
    mlir::GreedyRewriteConfig config;
    config.setListener(compile_analysis::patternListener());
    if (mlir::failed(
            mlir::applyPatternsGreedily(demo.get(), std::move(patterns), config))) {
      llvm::errs() << "iree-harness: demo rewrite failed\n";
      return 1;
    }
    llvm::outs() << "=== demo module after greedy rewrite ===\n";
    demo->print(llvm::outs());
    llvm::outs() << "\n";
  }
  return 0;
}
