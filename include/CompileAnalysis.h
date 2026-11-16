//===- CompileAnalysis.h - per-action compile time trace --------*- C++ -*-===//
//
// Read-only MLIR action handler that writes every dispatched action as a
// Chrome Trace Event, together with the identity of the operation it ran on
// and, for pass executions, the operation census of the pass scope before and
// after it ran.
//
// A `RewriterBase::Listener` that can be handed to the greedy and dialect
// conversion drivers records the same way: one event pair per pattern
// attempt, carrying the outcome, the reported match-failure reason and the
// number of operations the attempt inserted, modified, replaced and erased.
//
// Design and coverage boundaries: README.md.
// Aggregation: tools/ca-report.py.
//
//===----------------------------------------------------------------------===//

#ifndef COMPILE_ANALYSIS_COMPILEANALYSIS_H
#define COMPILE_ANALYSIS_COMPILEANALYSIS_H

#include "mlir/IR/Action.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace compile_analysis {

//===----------------------------------------------------------------------===//
// Options
//===----------------------------------------------------------------------===//

/// Collector configuration. The environment form is documented in README.md.
struct Options {
  /// Trace output path; "-" writes to stderr, empty disables the collector.
  std::string path;

  /// Walk the pass scope before and after each pass execution to record the
  /// operation count and the per-kind distribution. Costs one O(n) walk per
  /// pass execution per side, where n is the number of operations in the
  /// scope, so it is enabled by default but can be turned off.
  bool countOperations = true;

  /// Upper bound on the distinct operation kinds written per event. Kinds
  /// beyond the bound are dropped and the event is marked truncated.
  unsigned maxKinds = 64;

  /// Truncation of the pattern name in a pattern-attempt event. Some patterns
  /// are anonymous and have no name at all; those keep an empty name.
  size_t nameLimit = 96;

  /// Truncation of a match-failure reason. The reason is a full diagnostic in
  /// the worst case, with types and operands inlined.
  size_t reasonLimit = 160;

  /// Record pattern attempts and their IR changes. Requires handing
  /// `patternListener()` to the rewrite driver; the listener itself is the
  /// cost, so this switch only affects what the report can show.
  bool countPatterns = true;
};

/// Options read from the environment: COMPILE_ANALYSIS holds the trace path,
/// COMPILE_ANALYSIS_OPS=0 disables the operation census.
inline Options optionsFromEnvironment() {
  Options options;
  if (const char *path = std::getenv("COMPILE_ANALYSIS"))
    options.path = path;
  if (const char *ops = std::getenv("COMPILE_ANALYSIS_OPS"))
    options.countOperations = llvm::StringRef(ops) != "0";
  if (const char *maxKinds = std::getenv("COMPILE_ANALYSIS_MAX_KINDS")) {
    unsigned parsed = 0;
    if (!llvm::StringRef(maxKinds).getAsInteger(10, parsed))
      options.maxKinds = parsed;
  }
  if (const char *patterns = std::getenv("COMPILE_ANALYSIS_PATTERNS"))
    options.countPatterns = llvm::StringRef(patterns) != "0";
  return options;
}

//===----------------------------------------------------------------------===//
// Collector
//===----------------------------------------------------------------------===//

/// Chrome Trace writer for MLIR actions.
///
/// One instance is shared by every context in the process, so all of them
/// share one file and one time origin. Only the stream write is serialized;
/// timestamps and the operation census happen outside the lock, so concurrent
/// compilation is not driven into the lock.
class Profiler {
public:
  explicit Profiler(Options options)
      : options(std::move(options)), start(std::chrono::steady_clock::now()) {
    // No path means the collector is off: stay inactive without touching the
    // filesystem.
    if (this->options.path.empty())
      return;
    if (this->options.path == "-") {
      os = &llvm::errs();
      openArray();
      return;
    }
    std::error_code errorCode;
    auto file = std::make_unique<llvm::raw_fd_ostream>(this->options.path,
                                                      errorCode);
    if (errorCode) {
      llvm::errs() << "COMPILE_ANALYSIS: cannot open '" << this->options.path
                   << "': " << errorCode.message() << "\n";
      return;
    }
    owned = std::move(file);
    os = owned.get();
    openArray();
  }

  ~Profiler() {
    if (!os)
      return;
    std::lock_guard<std::mutex> guard(mutex);
    if (!first)
      *os << "\n";
    *os << "]\n";
    os->flush();
  }

  Profiler(const Profiler &) = delete;
  Profiler &operator=(const Profiler &) = delete;

  /// False when the trace file could not be opened.
  bool isActive() const { return os != nullptr; }

  /// One `args` object of an event. Handles the separators and the escaping so
  /// that callers never write raw text into the stream.
  class Fields {
  public:
    explicit Fields(llvm::raw_ostream &os) : os(os) {}

    Fields &text(llvm::StringRef key, llvm::StringRef value) {
      open(key);
      os << '"';
      escape(os, value);
      os << '"';
      return *this;
    }

    Fields &number(llvm::StringRef key, long long value) {
      open(key);
      os << value;
      return *this;
    }

    /// Inserts `raw` as the value of `key`; the caller owns its validity.
    Fields &json(llvm::StringRef key, llvm::StringRef raw) {
      open(key);
      os << raw;
      return *this;
    }

    /// Escapes for JSON, keeping the result valid UTF-8: operation names are
  /// ASCII, but a location carries whatever bytes the source file had, and a
  /// single stray byte makes the whole trace unparseable.
    static void escape(llvm::raw_ostream &os, llvm::StringRef value) {
      const auto *bytes = reinterpret_cast<const unsigned char *>(value.data());
      const size_t size = value.size();
      for (size_t index = 0; index < size; ++index) {
        const unsigned char character = bytes[index];
        switch (character) {
        case '"':
          os << "\\\"";
          continue;
        case '\\':
          os << "\\\\";
          continue;
        case '\n':
          os << "\\n";
          continue;
        case '\r':
          os << "\\r";
          continue;
        case '\t':
          os << "\\t";
          continue;
        default:
          break;
        }
        if (character < 0x20) {
          writeEscapedByte(os, character);
          continue;
        }
        if (character < 0x80) {
          os << static_cast<char>(character);
          continue;
        }
        // A multi-byte sequence is copied whole when it is well formed, and
        // escaped byte by byte when it is not.
        const size_t length = utf8SequenceLength(character);
        if (length > 1 && index + length <= size) {
          size_t offset = 1;
          for (; offset < length; ++offset) {
            if ((bytes[index + offset] & 0xC0) != 0x80)
              break;
          }
          if (offset == length) {
            os.write(value.data() + index, length);
            index += length - 1;
            continue;
          }
        }
        writeEscapedByte(os, character);
      }
    }

    /// Number of bytes a UTF-8 sequence starting with `lead` occupies, 0 when
    /// `lead` cannot start one.
    static size_t utf8SequenceLength(unsigned char lead) {
      if (lead >= 0xC2 && lead <= 0xDF)
        return 2;
      if (lead >= 0xE0 && lead <= 0xEF)
        return 3;
      if (lead >= 0xF0 && lead <= 0xF4)
        return 4;
      return 0;
    }

    /// Writes one byte as a `\u00XX` escape, which JSON accepts and which
    /// keeps the original value readable in the trace.
    static void writeEscapedByte(llvm::raw_ostream &os, unsigned char byte) {
      static const char digits[] = "0123456789abcdef";
      os << "\\u00" << digits[(byte >> 4) & 0xF] << digits[byte & 0xF];
    }

  private:
    void open(llvm::StringRef key) {
      if (!first)
        os << ',';
      first = false;
      os << '"';
      escape(os, key);
      os << "\":";
    }

    llvm::raw_ostream &os;
    bool first = true;
  };

  /// Write one event. `writeArgs` fills the `args` object; it runs under the
  /// stream lock, so it must not call back into `record`. Thread safe.
  void record(llvm::StringRef tag, llvm::StringRef phase,
              std::chrono::steady_clock::time_point timestamp,
              llvm::function_ref<void(Fields &)> writeArgs) {
    if (!os)
      return;
    // Integer microseconds: a `double` printed by the default raw_ostream
    // format keeps only 7 significant digits, which quantizes every timestamp
    // past 10 s onto a 1 ms grid.
    const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
                            timestamp - start)
                            .count();
    std::lock_guard<std::mutex> guard(mutex);
    if (!first)
      *os << ",\n";
    first = false;
    *os << R"({"name":")";
    Fields::escape(*os, tag);
    *os << R"(","cat":"compile-analysis","ph":")" << phase
        << R"(","pid":0,"tid":)" << llvm::get_threadid() << R"(,"ts":)"
        << micros << R"(,"args":{)";
    Fields fields(*os);
    writeArgs(fields);
    // Close `args` and the event itself.
    *os << "}}";
  }

  /// Bracket the transform with two events. Identity is captured before the
  /// transform runs: a pass may erase the operation the action refers to.
  ///
  /// `nested` is a handler that was already installed on the context, if any;
  /// it is called in place of the transform, so its own instrumentation ends
  /// up inside this event rather than around it.
  void operator()(llvm::function_ref<void()> transform,
                  const mlir::tracing::Action &action,
                  const mlir::MLIRContext::HandlerTy &nested = {}) {
    if (!os) {
      if (nested)
        nested(transform, action);
      else
        transform();
      return;
    }
    const Identity identity(action);
    const auto begin = std::chrono::steady_clock::now();
    emit(action, "B", begin, identity, census(identity.root));
    if (nested)
      nested(transform, action);
    else
      transform();
    const auto end = std::chrono::steady_clock::now();
    // MLIR runs the verifier on the same operation pointer after the pass has
    // returned (mlir/lib/Pass/Pass.cpp), so the anchor operation is still
    // alive here for every pass that the framework accepts.
    emit(action, "E", end, identity, census(identity.root));
  }

  /// Record an instant event naming a pipeline stage. A trace keeps the action
  /// stream only, which does not say where one pipeline ends and the next
  /// begins; the driver marks each stage so that the report can attribute every
  /// pass execution without guessing from pass names.
  void mark(llvm::StringRef stage) {
    if (!os)
      return;
    record("stage", "i", std::chrono::steady_clock::now(),
           [&](Fields &fields) { fields.text("stage", stage); });
  }

  /// Install on `context`, keeping a handler that is already installed by
  /// wrapping it: the previous handler still sees every action, with its
  /// transform bracketed by this collector.
  bool attach(mlir::MLIRContext &context) {
    if (!os)
      return false;
    std::lock_guard<std::mutex> guard(installMutex);
    if (!attached.insert(&context).second)
      return true;
    Profiler *self = this;
    mlir::MLIRContext::HandlerTy previous = context.getActionHandler();
    context.registerActionHandler(
        [self, previous](llvm::function_ref<void()> transform,
                         const mlir::tracing::Action &action) {
          (*self)(transform, action, previous);
        });
    return true;
  }

  /// Options in force; read by the rewrite-driver listener for its limits.
  const Options &getOptions() const { return options; }

private:
  /// What the action refers to, read before the transform runs.
  struct Identity {
    mlir::Operation *root = nullptr;
    llvm::SmallString<128> description;
    llvm::StringRef passName;
    llvm::StringRef passArgument;
    llvm::StringRef opName;
    llvm::SmallString<64> symbol;
    llvm::SmallString<160> location;
    bool isPassExecution = false;

    explicit Identity(const mlir::tracing::Action &action) {
      description.clear();
      llvm::raw_svector_ostream stream(description);
      action.print(stream);
      // Implementations may end with a newline. Truncate in place: assigning a
      // StringRef that views this very buffer to the SmallString trips
      // SmallVector's own reference-invalidation assertion.
      description.truncate(llvm::StringRef(description).rtrim().size());

      if (auto *passAction =
              llvm::dyn_cast<mlir::PassExecutionAction>(&action)) {
        isPassExecution = true;
        const mlir::Pass &pass = passAction->getPass();
        passName = pass.getName();
        passArgument = pass.getArgument();
      }

      for (mlir::IRUnit unit : action.getContextIRUnits()) {
        auto *op = llvm::dyn_cast_if_present<mlir::Operation *>(unit);
        if (!op)
          continue;
        root = op;
        opName = op->getName().getStringRef();
        if (auto symbol = op->getAttrOfType<mlir::StringAttr>("sym_name"))
          this->symbol = symbol.getValue();
        if (auto fileLoc = llvm::dyn_cast<mlir::FileLineColLoc>(op->getLoc())) {
          llvm::raw_svector_ostream loc(location);
          loc << fileLoc.getFilename() << ':' << fileLoc.getLine() << ':'
              << fileLoc.getColumn();
        } else if (auto nameLoc = llvm::dyn_cast<mlir::NameLoc>(op->getLoc())) {
          location = nameLoc.getName().getValue();
        }
        break;
      }
    }
  };

  /// Operation count of the scope rooted at `root`, plus the per-kind
  /// distribution. Empty when disabled or when the action carries no
  /// operation of its own.
  struct Census {
    bool present = false;
    unsigned total = 0;
    unsigned distinctKinds = 0;
    bool truncated = false;
    // Owns the names: the counts map below does not outlive its lookup, so a
    // StringRef into it dangles by the time the event is written.
    std::vector<std::pair<std::string, unsigned>> kinds;
  };

  Census census(mlir::Operation *root) const {
    Census census;
    if (!options.countOperations || !root)
      return census;
    census.present = true;
    llvm::StringMap<unsigned> counts;
    root->walk([&](mlir::Operation *op) {
      ++census.total;
      ++counts[op->getName().getStringRef()];
    });
    census.distinctKinds = counts.size();
    census.kinds.reserve(counts.size());
    for (const auto &entry : counts)
      census.kinds.emplace_back(entry.getKey().str(), entry.getValue());
    // A stable order keeps two traces of the same compile diffable; the
    // report aggregates counts, not order.
    std::sort(census.kinds.begin(), census.kinds.end(),
              [](const auto &left, const auto &right) {
                if (left.second != right.second)
                  return left.second > right.second;
                return left.first < right.first;
              });
    if (census.kinds.size() > options.maxKinds) {
      census.kinds.resize(options.maxKinds);
      census.truncated = true;
    }
    return census;
  }

  /// `[` opens the array; every event is written on its own line so readers
  /// can stream the file instead of loading it whole.
  void openArray() { *os << "[\n"; }

  void emit(const mlir::tracing::Action &action, llvm::StringRef phase,
            std::chrono::steady_clock::time_point timestamp,
            const Identity &identity, const Census &census) {
    // Identity travels on the opening event only: a rewrite may erase the
    // operation before the closing event is written.
    const bool opening = phase == "B";
    record(action.getTag(), phase, timestamp, [&](Fields &fields) {
      fields.text("desc", identity.description);
      if (opening) {
        if (!identity.passName.empty())
          fields.text("pass", identity.passName);
        if (!identity.passArgument.empty())
          fields.text("arg", identity.passArgument);
        if (!identity.opName.empty())
          fields.text("op", identity.opName);
        if (!identity.symbol.empty())
          fields.text("sym", identity.symbol);
        if (!identity.location.empty())
          fields.text("loc", identity.location);
      }
      if (census.present && identity.isPassExecution)
        writeCensus(fields, opening ? "opsBefore" : "opsAfter", census);
    });
  }

  void writeCensus(Fields &fields, llvm::StringRef prefix,
                   const Census &census) {
    fields.number(prefix, census.total);
    llvm::SmallString<512> kinds;
    llvm::raw_svector_ostream kindsStream(kinds);
    kindsStream << '{';
    for (size_t index = 0; index < census.kinds.size(); ++index) {
      if (index)
        kindsStream << ',';
      kindsStream << '"';
      Fields::escape(kindsStream, census.kinds[index].first);
      kindsStream << "\":" << census.kinds[index].second;
    }
    kindsStream << '}';
    llvm::SmallString<64> kindsKey(prefix);
    kindsKey += "Kinds";
    fields.json(kindsKey, kinds);
    if (census.truncated) {
      llvm::SmallString<64> droppedKey(prefix);
      droppedKey += "KindsDropped";
      fields.number(droppedKey, census.distinctKinds - census.kinds.size());
    }
  }

  Options options;
  std::chrono::steady_clock::time_point start;
  std::mutex mutex;
  std::unique_ptr<llvm::raw_fd_ostream> owned;
  llvm::raw_ostream *os = nullptr;
  bool first = true;

  std::mutex installMutex;
  std::set<const void *> attached;
};

//===----------------------------------------------------------------------===//
// Rewrite-driver listener
//===----------------------------------------------------------------------===//

/// Records every pattern attempt of a rewrite driver that is handed this
/// listener: the pattern name, the root operation, the outcome, the
/// match-failure reason the pattern reported, and how many operations the
/// attempt inserted, modified, replaced and erased.
///
/// `notifyPatternBegin` / `notifyPatternEnd` bracket one `matchAndRewrite`
/// call, so the pair also yields the duration of a single attempt, which the
/// action stream cannot express (an action carries no outcome).
///
/// Attaching this listener is not free: the greedy driver only passes its
/// onSuccess / onFailure callbacks to the pattern applicator when a listener
/// is set (mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp), so the
/// attempt path runs with more callbacks and two clock reads per attempt.
class PatternListener : public mlir::RewriterBase::Listener {
public:
  explicit PatternListener(Profiler &profiler) : profiler(profiler) {}

  void notifyPatternBegin(const mlir::Pattern &pattern,
                          mlir::Operation *op) override {
    attempts().emplace_back();
    Attempt &attempt = attempts().back();
    attempt.patternName = llvm::StringRef(pattern.getDebugName()).substr(
        0, profiler.getOptions().nameLimit);
    if (op)
      attempt.opName = op->getName().getStringRef();
    attempt.begin = std::chrono::steady_clock::now();
    profiler.record("pattern-attempt", "B", attempt.begin,
                    [&](Profiler::Fields &fields) {
                      fields.text("pattern", attempt.patternName);
                      if (!attempt.opName.empty())
                        fields.text("op", attempt.opName);
                    });
  }

  void notifyPatternEnd(const mlir::Pattern &pattern,
                        mlir::LogicalResult status) override {
    // A begin without an end (or an end without a begin) means the driver was
    // interrupted; dropping the entry keeps the stack paired.
    if (attempts().empty())
      return;
    Attempt attempt = std::move(attempts().back());
    attempts().pop_back();
    const bool succeeded = mlir::succeeded(status);
    profiler.record(
        "pattern-attempt", "E", std::chrono::steady_clock::now(),
        [&](Profiler::Fields &fields) {
          fields.text("pattern", attempt.patternName);
          if (!attempt.opName.empty())
            fields.text("op", attempt.opName);
          fields.text("outcome", succeeded ? "success" : "failure");
          if (!attempt.reason.empty())
            fields.text("reason", attempt.reason);
          if (attempt.inserted || attempt.modified || attempt.replaced ||
              attempt.erased) {
            fields.number("inserted", attempt.inserted);
            fields.number("modified", attempt.modified);
            fields.number("replaced", attempt.replaced);
            fields.number("erased", attempt.erased);
          }
        });
  }

  void notifyMatchFailure(
      mlir::Location loc,
      llvm::function_ref<void(mlir::Diagnostic &)> reasonCallback) override {
    // Only the first reason is kept: a pattern may report several reasons for
    // one attempt, and the count of reasons is not the count of attempts.
    if (attempts().empty() || !attempts().back().reason.empty())
      return;
    mlir::Diagnostic diagnostic(loc, mlir::DiagnosticSeverity::Remark);
    reasonCallback(diagnostic);
    Attempt &attempt = attempts().back();
    // `str()` returns a temporary; the StringRef has to view this copy.
    const std::string text = diagnostic.str();
    const llvm::StringRef reason(text);
    const size_t limit = profiler.getOptions().reasonLimit;
    attempt.reason =
        llvm::StringRef(reason.begin(), std::min(reason.size(), limit));
    // The reason is a diagnostic; a newline would break one event per line.
    for (char &character : attempt.reason)
      if (character == '\n' || character == '\r')
        character = ' ';
  }

  void notifyOperationInserted(mlir::Operation *op,
                               mlir::OpBuilder::InsertPoint previous) override {
    countIfInAttempt(&Attempt::inserted);
  }

  void notifyOperationModified(mlir::Operation *op) override {
    countIfInAttempt(&Attempt::modified);
  }

  void notifyOperationReplaced(mlir::Operation *op,
                               mlir::ValueRange replacement) override {
    countIfInAttempt(&Attempt::replaced);
  }

  void notifyOperationErased(mlir::Operation *op) override {
    countIfInAttempt(&Attempt::erased);
  }

private:
  struct Attempt {
    llvm::SmallString<96> patternName;
    llvm::SmallString<64> opName;
    llvm::SmallString<160> reason;
    std::chrono::steady_clock::time_point begin;
    unsigned inserted = 0;
    unsigned modified = 0;
    unsigned replaced = 0;
    unsigned erased = 0;
  };

  /// One stack per thread: drivers run concurrently, and the callbacks of one
  /// attempt always arrive on the thread that began it.
  static std::vector<Attempt> &attempts() {
    static thread_local std::vector<Attempt> storage;
    return storage;
  }

  void countIfInAttempt(unsigned Attempt::*field) {
    // IR changes outside a pattern attempt come from the driver itself, not
    // from a pattern, so they are not attributed here.
    if (attempts().empty())
      return;
    ++(attempts().back().*field);
  }

  Profiler &profiler;
};

//===----------------------------------------------------------------------===//
// Process-wide runtime
//===----------------------------------------------------------------------===//

/// The collector and its listener share one lifetime and one trace file.
struct Runtime {
  explicit Runtime(Options options)
      : profiler(std::move(options)), listener(profiler) {}

  Profiler profiler;
  PatternListener listener;
};

/// Null when the collector is disabled; the collector itself is created on
/// the first call, so every entry point below shares one instance.
inline Runtime *runtime() {
  static Runtime instance(optionsFromEnvironment());
  if (!instance.profiler.isActive())
    return nullptr;
  return &instance;
}

//===----------------------------------------------------------------------===//
// Convenience entry points
//===----------------------------------------------------------------------===//

/// Install the collector on `context` when COMPILE_ANALYSIS names an output
/// path. Safe to call for every context: the first call opens the trace file,
/// later calls share it.
inline bool installIfRequested(mlir::MLIRContext &context) {
  Runtime *instance = runtime();
  if (!instance)
    return false;
  return instance->profiler.attach(context);
}

/// Mark the start of a pipeline stage in the trace. No-op when the collector is
/// off, so a driver calls it unconditionally:
///
///   compile_analysis::markStage("flow");
///
inline void markStage(llvm::StringRef stage) {
  if (Runtime *instance = runtime())
    instance->profiler.mark(stage);
}

/// Listener to hand to a rewrite driver, or null when the collector is off:
///
///   mlir::GreedyRewriteConfig config;
///   config.setListener(compile_analysis::patternListener());
///
///   mlir::ConversionConfig conversion;
///   conversion.listener = compile_analysis::patternListener();
///
/// Both setters accept a null pointer, so the call site needs no branch.
inline mlir::RewriterBase::Listener *patternListener() {
  Runtime *instance = runtime();
  if (!instance || !instance->profiler.getOptions().countPatterns)
    return nullptr;
  return &instance->listener;
}

/// Dialect registry extension; runs when the attached context is constructed,
/// i.e. before parsing and before any pass.
inline void installOnContext(mlir::MLIRContext *context) {
  installIfRequested(*context);
}

} // namespace compile_analysis

#endif // COMPILE_ANALYSIS_COMPILEANALYSIS_H
