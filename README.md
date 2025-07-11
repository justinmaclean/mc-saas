# SaaS Monte Carlo Analysis Tool

A comprehensive Streamlit application for performing Monte Carlo simulations on SaaS product launches.

## Features

- **Monte Carlo Simulation**: Run thousands of simulations with configurable parameters
- **Statistical Distributions**: Uses appropriate distributions for each parameter type
- **Financial Analysis**: Calculate NPV and ARR with proper discounting
- **Risk Assessment**: Analyze probabilities of different outcome scenarios
- **Interactive Visualizations**: Histograms and statistical summaries

## Installation

1. Create and activate a Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run monte_carlo_saas.py
```

## Parameters

### Customer Acquisition
- **Customer Acquisition Cost**: Cost to acquire each new customer
- **Monthly New Customers**: Base number of customers acquired per month
- **Monthly Growth Rate**: Growth rate of customer acquisition over time

### Revenue
- **Monthly Recurring Revenue**: Revenue per customer per month
- **Annual Churn Rate**: Percentage of customers lost per year

### Costs
- **Monthly Fixed Costs**: Fixed operational costs per month
- **Monthly Support Cost**: Support cost per customer per month
- **Initial R&D Cost**: One-time development cost
- **Monthly Technology Cost**: Ongoing technology expenses

Each parameter requires both an expected value and standard deviation for the Monte Carlo simulation.

## Output

The tool provides:
- NPV and ARR distribution histograms
- Summary statistics (mean, standard deviation)
- Percentile analysis (5th, 25th, 50th, 75th, 95th)
- Risk metrics (probability of negative NPV, etc.)

## Model Assumptions

- 5-year analysis horizon (60 months)
- Monthly application of churn rate
- Customer acquisition growth over time
- Proper NPV discounting (default 12% annual rate)
- Starting with 0 customers 