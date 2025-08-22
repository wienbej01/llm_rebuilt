.PHONY: help setup install lint type test fmt clean build docs serve setup-dev pre-commit install-pre-commit

# Default target
help:
	@echo "Available commands:"
	@echo "  setup         - Initial project setup and dependency installation"
	@echo "  install       - Install project dependencies"
	@echo "  lint          - Run ruff linter"
	@echo "  type          - Run mypy type checker"
	@echo "  test          - Run test suite"
	@echo "  fmt           - Format code with black and ruff"
	@echo "  clean         - Clean up cache files and build artifacts"
	@echo "  build         - Build package with hatch"
	@echo "  docs          - Build documentation"
	@echo "  serve         - Serve documentation locally"
	@echo "  setup-dev     - Setup development environment"
	@echo "  pre-commit    - Run pre-commit hooks"
	@echo "  install-pre-commit - Install pre-commit hooks"
	@echo ""
	@echo "Trading-specific commands:"
	@echo "  ingest        - Run data ingestion pipeline"
	@echo "  build-features- Build feature store"
	@echo "  backtest      - Run backtesting"
	@echo "  paper         - Run paper trading simulation"
	@echo "  validate-llm  - Validate LLM contracts"

# Setup and installation
setup: install install-pre-commit
	@echo "✅ Project setup complete!"

install:
	@echo "📦 Installing dependencies..."
	@pip install -e .
	@pip install -e ".[dev,docs]"
	@echo "✅ Dependencies installed!"

install-pre-commit:
	@echo "🔧 Installing pre-commit hooks..."
	@pre-commit install
	@echo "✅ Pre-commit hooks installed!"

setup-dev: setup
	@echo "🚀 Setting up development environment..."
	@mkdir -p logs data outputs configs
	@cp configs/settings.example.yaml configs/settings.yaml
	@echo "✅ Development environment ready!"

# Code quality
lint:
	@echo "🔍 Running ruff linter..."
	@ruff check .
	@echo "✅ Linting complete!"

type:
	@echo "🔍 Running mypy type checker..."
	@mypy .
	@echo "✅ Type checking complete!"

fmt:
	@echo "🎨 Formatting code..."
	@black .
	@ruff check --fix .
	@echo "✅ Code formatting complete!"

pre-commit:
	@echo "🔧 Running pre-commit hooks..."
	@pre-commit run --all-files
	@echo "✅ Pre-commit checks complete!"

# Testing
test:
	@echo "🧪 Running test suite..."
	@pytest -v
	@echo "✅ Tests complete!"

test-cov:
	@echo "🧪 Running tests with coverage..."
	@pytest --cov=pse_llm --cov-report=html --cov-report=term
	@echo "✅ Tests with coverage complete!"

test-unit:
	@echo "🧪 Running unit tests..."
	@pytest -m "unit" -v
	@echo "✅ Unit tests complete!"

test-integration:
	@echo "🧪 Running integration tests..."
	@pytest -m "integration" -v
	@echo "✅ Integration tests complete!"

# Building and packaging
build:
	@echo "🏗️ Building package..."
	@hatch build
	@echo "✅ Package built!"

clean:
	@echo "🧹 Cleaning up..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.pyd" -delete
	@find . -type f -name ".coverage" -delete
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# Documentation
docs:
	@echo "📚 Building documentation..."
	@mkdocs build
	@echo "✅ Documentation built!"

serve:
	@echo "🌐 Serving documentation locally..."
	@mkdocs serve

# Trading-specific commands
ingest:
	@echo "📊 Running data ingestion..."
	@python -m cli.main ingest --config configs/settings.yaml

build-features:
	@echo "🔧 Building feature store..."
	@python -m cli.main build-features --config configs/settings.yaml

backtest:
	@echo "📈 Running backtesting..."
	@python -m cli.main backtest --config configs/settings.yaml

paper:
	@echo "📊 Running paper trading..."
	@python -m cli.main paper --config configs/settings.yaml

validate-llm:
	@echo "🤖 Validating LLM contracts..."
	@python -m cli.main validate-llm --config configs/settings.yaml

# Development workflow
check-all: lint type test
	@echo "✅ All checks passed!"

# Docker (optional)
docker-build:
	@echo "🐳 Building Docker image..."
	@docker build -t pse-llm .
	@echo "✅ Docker image built!"

docker-run:
	@echo "🐳 Running Docker container..."
	@docker run --rm -it pse-llm

# CI/CD simulation
ci: clean check-all build
	@echo "✅ CI pipeline complete!"