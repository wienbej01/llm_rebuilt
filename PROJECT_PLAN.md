# PSE-LLM Rebuild & Orchestration Plan

## Overview
Rebuild the partially corrupted `pse-llm` repo to match the target system architecture. The current repo has good foundations but needs reorganization, completion of missing components, and implementation of orchestration for automated execution.

## Phases

### Phase 1: Bootstrap & Safety
**Goal**: Set up environment, quarantine corrupted files, and establish logging.

**Steps**:
1. **BOOT-ENV**: Create virtual environment, install dependencies, init logs/
   - Acceptance: venv active, ruff/mypy/pytest available
   - Owner: CodeMode
   - Deps: None
   - Effort: Low

2. **REPO-AUDIT**: Run audit script to inventory files, detect corruption
   - Acceptance: logs/repo_audit.json and .md produced
   - Owner: CodeMode
   - Deps: BOOT-ENV
   - Effort: Medium

3. **QUARANTINE**: Move corrupted files to wip/recovery/ with .bak
   - Acceptance: All corrupted files quarantined, recovery log updated
   - Owner: CodeMode
   - Deps: REPO-AUDIT
   - Effort: Low

4. **DELTA-ANALYSIS**: Compare current vs target, identify gaps
   - Acceptance: logs/delta_plan_vs_repo.md complete
   - Owner: Architect
   - Deps: REPO-AUDIT
   - Effort: Low

### Phase 2: Data Layer
**Goal**: Implement unified data interfaces, loaders, validators, and no-look-ahead guards.

**Steps**:
1. **DATA-INTERFACES**: Create IDataSource, IDataValidator interfaces
   - Acceptance: Interfaces defined with proper typing
   - Owner: CodeMode
   - Deps: BOOT-ENV
   - Effort: Medium

2. **DATA-LOADERS**: Implement all loaders (Databento local/API, Polygon, CSV)
   - Acceptance: All loaders working, unified interface
   - Owner: CodeMode
   - Deps: DATA-INTERFACES
   - Effort: High

3. **DATA-PIPELINE**: Build pipeline with features and validators
   - Acceptance: Pipeline processes data with no-look-ahead guards
   - Owner: CodeMode
   - Deps: DATA-LOADERS
   - Effort: High

4. **FEATURES**: Implement all required features (ATR, VWAP, RSI, etc.)
   - Acceptance: Features guarantee valid outputs, no NaN on short windows
   - Owner: CodeMode
   - Deps: DATA-PIPELINE
   - Effort: Medium

### Phase 3: Strategy Layer
**Goal**: Complete strategy base, implement all tactics and exits.

**Steps**:
1. **STRATEGY-BASE**: Enhance StrategyBase with TacticBase, Signal objects, Rule DSL
   - Acceptance: Base classes support composition and rule evaluation
   - Owner: CodeMode
   - Deps: DATA-LAYER
   - Effort: Medium

2. **TACTICS**: Implement all tactics (ORB, Momentum, Mean Reversion, ATR Channel, VWAP Reversion, S/R Break Retest, ICT Killzone, FVG)
   - Acceptance: All tactics detect setups correctly, deterministic rule evaluation
   - Owner: CodeMode
   - Deps: STRATEGY-BASE
   - Effort: High

3. **EXITS**: Implement all exit strategies (RR Static, ATR Trail, VWAP Trail, Time Stop, Break Even)
   - Acceptance: Exits fire correctly, support breakeven flips
   - Owner: CodeMode
   - Deps: STRATEGY-BASE
   - Effort: Medium

4. **PORTFOLIO**: Build multi-tactic aggregation and conflict resolution
   - Acceptance: Portfolio handles overlapping signals, prioritizes by quality
   - Owner: CodeMode
   - Deps: TACTICS, EXITS
   - Effort: Medium

### Phase 4: Risk Layer
**Goal**: Implement position sizing, limits, drawdown, and compliance.

**Steps**:
1. **SIZING**: Build sizing logic (fixed, vol-adjusted)
   - Acceptance: Sizing respects account risk limits
   - Owner: CodeMode
   - Deps: STRATEGY-LAYER
   - Effort: Medium

2. **LIMITS**: Implement position limits (max positions, heat, instrument caps)
   - Acceptance: Limits enforced per trade and aggregate
   - Owner: CodeMode
   - Deps: SIZING
   - Effort: Low

3. **DRAWDOWN**: Add MDD tracking and halt logic
   - Acceptance: MDD halts trading when threshold reached
   - Owner: CodeMode
   - Deps: LIMITS
   - Effort: Medium

4. **COMPLIANCE**: Build sanity guards and risk checks
   - Acceptance: Guards prevent invalid orders
   - Owner: CodeMode
   - Deps: DRAWDOWN
   - Effort: Low

### Phase 5: Backtest Layer
**Goal**: Complete backtester with engine, analyzers, metrics, audit.

**Steps**:
1. **BACKTEST-ENGINE**: Implement backtest engine with no-look-ahead
   - Acceptance: Engine processes bars correctly, enforces no-leak
   - Owner: CodeMode
   - Deps: RISK-LAYER
   - Effort: High

2. **ANALYZERS**: Build analyzers for trade analysis
   - Acceptance: Analyzers produce accurate statistics
   - Owner: CodeMode
   - Deps: BACKTEST-ENGINE
   - Effort: Medium

3. **METRICS**: Implement metrics calculation (Sharpe, MDD, etc.)
   - Acceptance: Metrics reconcile with ledger
   - Owner: CodeMode
   - Deps: ANALYZERS
   - Effort: Medium

4. **AUDIT**: Add forensic bar-by-bar audit
   - Acceptance: Audit logs every decision with reasons
   - Owner: CodeMode
   - Deps: METRICS
   - Effort: Medium

5. **FIXTURES**: Create sample datasets for testing
   - Acceptance: Fixtures cover edge cases
   - Owner: CodeMode
   - Deps: AUDIT
   - Effort: Low

### Phase 6: LLM Layer
**Goal**: Complete LLM integration with memory, guards, and all agents.

**Steps**:
1. **LLM-INTERFACE**: Enhance interface with rate limits and async
   - Acceptance: Interface handles all providers reliably
   - Owner: CodeMode
   - Deps: BACKTEST-LAYER
   - Effort: Low

2. **VALIDATOR**: Complete PSE validator with risk constraints
   - Acceptance: Validator approves/denies with rationale
   - Owner: CodeMode
   - Deps: LLM-INTERFACE
   - Effort: Medium

3. **PROPOSER**: Implement holistic proposer
   - Acceptance: Proposer generates valid trades
   - Owner: CodeMode
   - Deps: VALIDATOR
   - Effort: Medium

4. **MEMORY**: Add minimal state memory
   - Acceptance: Memory tracks constraints without full payloads
   - Owner: CodeMode
   - Deps: PROPOSER
   - Effort: Low

5. **GUARDS**: Implement token and privacy guards
   - Acceptance: Guards prevent leaks and overuse
   - Owner: CodeMode
   - Deps: MEMORY
   - Effort: Low

### Phase 7: Paper Trading Layer
**Goal**: Implement IBKR paper trading with client, router, driver.

**Steps**:
1. **IBKR-CLIENT**: Build ib_insync wrapper
   - Acceptance: Client connects, places/cancels orders
   - Owner: CodeMode
   - Deps: LLM-LAYER
   - Effort: Medium

2. **ORDER-ROUTER**: Implement order routing logic
   - Acceptance: Router translates intents to broker orders
   - Owner: CodeMode
   - Deps: IBKR-CLIENT
   - Effort: Medium

3. **PAPER-DRIVER**: Create paper driver with mocks
   - Acceptance: Driver runs dry-run successfully
   - Owner: CodeMode
   - Deps: ORDER-ROUTER
   - Effort: Medium

### Phase 8: Testing & QA Gates
**Goal**: Build comprehensive tests and quality gates.

**Steps**:
1. **UNIT-TESTS**: Write unit tests for all components
   - Acceptance: >=95% coverage, all critical logic tested
   - Owner: CodeMode
   - Deps: PAPER-TRADING
   - Effort: High

2. **INTEGRATION-TESTS**: Build integration tests for full backtest
   - Acceptance: Tests pass invariants (no-look-ahead, reconciliation)
   - Owner: CodeMode
   - Deps: UNIT-TESTS
   - Effort: Medium

3. **E2E-TESTS**: Implement e2e tests for paper trading
   - Acceptance: Dry-run and smoke tests pass
   - Owner: CodeMode
   - Deps: INTEGRATION-TESTS
   - Effort: Medium

4. **QA-GATES**: Create QA gate system
   - Acceptance: Gates enforce acceptance criteria
   - Owner: CodeMode
   - Deps: E2E-TESTS
   - Effort: Low

### Phase 9: Docs & DX
**Goal**: Complete documentation and developer experience.

**Steps**:
1. **README**: Update README with quickstarts
   - Acceptance: README covers backtest and paper workflows
   - Owner: Architect
   - Deps: QA-GATES
   - Effort: Low

2. **SCRIPTS**: Create all utility scripts
   - Acceptance: Scripts for bootstrap, backtest, report generation
   - Owner: CodeMode
   - Deps: README
   - Effort: Low

3. **EXAMPLES**: Add example configurations and usage
   - Acceptance: Examples demonstrate key features
   - Owner: Architect
   - Deps: SCRIPTS
   - Effort: Low

## QA Gates
- **Lint**: ruff clean
- **Type Check**: mypy clean
- **Unit Tests**: >=95% pass
- **Integration Tests**: Pass no-look-ahead and reconciliation
- **E2E Tests**: Paper dry-run successful

## Dependencies
- All phases depend on previous phase completion
- Cross-phase deps noted in individual steps
- QA gates run after each phase

## Risk Mitigation
- Quarantine corrupted files before changes
- Backup before structural moves
- Test no-look-ahead property throughout
- Validate risk calculations
- Ensure deterministic behavior

## Success Criteria
- All modules implemented per target skeleton
- No-look-ahead enforced
- Backtest reconciles with ledger
- Paper trading ready for live smoke test
- Full test suite passing
- Orchestrator manages execution end-to-end