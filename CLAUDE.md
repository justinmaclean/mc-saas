# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a SaaS Monte Carlo Analysis Tool built with Streamlit. It provides Monte Carlo simulations for SaaS product launches, allowing users to model various business scenarios with uncertainty in key parameters.

## Key Application Architecture

### Single-File Application Structure
- **Main Application**: `monte_carlo_saas.py` - Contains the entire Streamlit application
- **Dependencies**: Managed through `requirements.txt`
- **Virtual Environment**: Uses `venv/` directory for Python environment isolation

### Core Components in monte_carlo_saas.py

1. **Parameter Input System (lines 52-158)**
   - Sidebar-based input forms for all simulation parameters
   - Supports 6 distribution types: Normal, Log-normal, Uniform, Triangular, Beta, Gamma
   - Each parameter has distribution type, mean, and standard deviation controls

2. **Distribution Sampling Engine (lines 159-215)**
   - `generate_samples_by_distribution()` - Converts user inputs to statistical distributions
   - `generate_random_samples()` - Orchestrates sampling for all parameters
   - Handles parameter validation and bounds checking

3. **Business Simulation Model (lines 311-390)**
   - `simulate_business()` - Core Monte Carlo simulation engine
   - Models 5-year business lifecycle with monthly granularity
   - Calculates NPV, ARR, CLV, customer growth, and cash flows
   - Applies churn, growth rates, and proper discounting

4. **Risk Management System (lines 494-655)**
   - Dynamic risk definition and application
   - Risk modal UI with custom CSS styling
   - Multiplicative risk impacts on base parameters
   - Session state management for risk persistence

5. **Visualization and Analysis (lines 658-1028)**
   - Multiple chart types: histograms, time series, percentile tables
   - Interactive Plotly charts with statistical overlays
   - CSV export functionality with comprehensive data structure

## Development Commands

### Running the Application
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit application
streamlit run monte_carlo_saas.py
```

### Development Workflow
```bash
# Check for Python syntax errors
python -m py_compile monte_carlo_saas.py

# Update dependencies
pip freeze > requirements.txt
```

## Key Business Logic

### Financial Modeling
- **NPV Calculation**: Monthly cash flows discounted at specified annual rate
- **Customer Churn**: Annual rate converted to monthly, applied cumulatively
- **Revenue Recognition**: Monthly recurring revenue multiplied by active customers
- **Cost Structure**: CAC (one-time), support costs (per customer), fixed costs (monthly)

### Statistical Distributions
- **Normal**: For symmetric parameters like costs and growth rates
- **Log-normal**: For positive-skewed values like revenue metrics
- **Beta**: For bounded percentages like churn rates (0-100%)
- **Gamma**: For positive count-like values like acquisition rates
- **Uniform/Triangular**: For scenario modeling with defined ranges

### Session State Management
The application heavily uses Streamlit's session state to:
- Cache simulation results for performance
- Manage auto-run functionality
- Persist risk definitions
- Store comprehensive simulation data for CSV export

## Important Implementation Details

### Auto-Run Feature
- Parameter change detection through hashing (lines 461-471)
- Automatic simulation triggering when inputs change
- Reduced simulation count (3,000) for responsive auto-run vs manual run (5,000)

### Risk System Architecture
- Modal-based risk definition UI with custom CSS
- Multiplicative risk application to base parameters
- Risk occurrence tracking per simulation for analysis
- Session state persistence of risk configurations

### Data Export Structure
- **Input Assumptions**: All random parameter values used per simulation
- **Output Results**: NPV, ARR, CLV, customer counts for each simulation
- **Annual Breakdowns**: Revenue and EBITDA by year for trend analysis
- **Risk Flags**: Which risks occurred in each simulation

## Common Modification Patterns

When adding new parameters:
1. Add UI controls in sidebar section
2. Update `generate_random_samples()` to include new parameter
3. Modify `simulate_business()` to incorporate parameter in calculations
4. Add parameter to CSV export structure
5. Update risk system parameter choices if applicable

When modifying distributions:
1. Update `generate_samples_by_distribution()` for new distribution logic
2. Ensure proper parameter validation and bounds checking
3. Update distribution guide in sidebar help text

## Dependencies

Core libraries and their purposes:
- `streamlit`: Web application framework
- `numpy`: Numerical computing and random sampling
- `pandas`: Data manipulation for CSV export
- `plotly`: Interactive charting and visualization
- `scipy`: Statistical distributions and calculations
- `matplotlib/seaborn`: Additional plotting capabilities
- `streamlit-modal`: Modal dialogs for risk management UI