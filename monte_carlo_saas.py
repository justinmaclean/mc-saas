import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm, lognorm, beta, gamma
from scipy import stats

# Set page config
st.set_page_config(
    page_title="SaaS Monte Carlo Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 SaaS Product Launch Monte Carlo Analysis")
st.markdown("---")

# Sidebar for inputs
st.sidebar.header("📋 Model Parameters")

# Simulation settings
st.sidebar.subheader("🎯 Simulation Settings")
auto_run = st.sidebar.checkbox("🔄 Auto-run simulation when parameters change", value=True)
num_simulations = st.sidebar.slider("Number of Simulations", 1000, 10000, 3000 if auto_run else 5000, 1000)
discount_rate = st.sidebar.slider("Discount Rate (%)", 5.0, 20.0, 12.0, 0.5) / 100

if auto_run:
    st.sidebar.info("💡 Simulation runs automatically when you change parameters")
else:
    st.sidebar.info("🎯 Click 'Run Simulation' button to start analysis")

st.sidebar.markdown("---")

# Customer Acquisition Parameters
st.sidebar.subheader("👥 Customer Acquisition")
col1, col2 = st.sidebar.columns(2)

with col1:
    st.write("**Expected Values**")
    cac_mean = st.number_input("Customer Acquisition Cost ($)", value=150.0, min_value=0.0)
    acquisition_rate_mean = st.number_input("Monthly New Customers", value=100.0, min_value=0.0)
    acquisition_growth_mean = st.number_input("Monthly Growth Rate (%)", value=2.0, min_value=-10.0, max_value=50.0) / 100

with col2:
    st.write("**Standard Deviation**")
    cac_std = st.number_input("CAC Std Dev ($)", value=30.0, min_value=0.1)
    acquisition_rate_std = st.number_input("Acquisition Rate Std Dev", value=20.0, min_value=0.1)
    acquisition_growth_std = st.number_input("Growth Rate Std Dev (%)", value=1.0, min_value=0.1, max_value=10.0) / 100

# Revenue Parameters
st.sidebar.subheader("💰 Revenue")
col1, col2 = st.sidebar.columns(2)

with col1:
    mrr_mean = st.number_input("Monthly Recurring Revenue per Customer ($)", value=50.0, min_value=0.0)
    churn_rate_mean = st.number_input("Annual Churn Rate (%)", value=15.0, min_value=0.0, max_value=100.0) / 100

with col2:
    mrr_std = st.number_input("MRR Std Dev ($)", value=10.0, min_value=0.1)
    churn_rate_std = st.number_input("Churn Rate Std Dev (%)", value=3.0, min_value=0.1, max_value=20.0) / 100

# Cost Parameters
st.sidebar.subheader("💸 Costs")
col1, col2 = st.sidebar.columns(2)

with col1:
    fixed_costs_mean = st.number_input("Monthly Fixed Costs ($)", value=10000.0, min_value=0.0)
    support_cost_mean = st.number_input("Monthly Support Cost per Customer ($)", value=5.0, min_value=0.0)
    rd_cost_mean = st.number_input("Initial R&D Cost ($)", value=100000.0, min_value=0.0)
    tech_cost_mean = st.number_input("Monthly Technology Cost ($)", value=2000.0, min_value=0.0)

with col2:
    fixed_costs_std = st.number_input("Fixed Costs Std Dev ($)", value=2000.0, min_value=0.1)
    support_cost_std = st.number_input("Support Cost Std Dev ($)", value=1.0, min_value=0.1)
    rd_cost_std = st.number_input("R&D Cost Std Dev ($)", value=20000.0, min_value=0.1)
    tech_cost_std = st.number_input("Tech Cost Std Dev ($)", value=500.0, min_value=0.1)

def generate_random_samples(num_sims):
    """Generate random samples for all parameters using appropriate distributions"""
    
    # Customer Acquisition Cost (Normal)
    cac_samples = np.random.normal(cac_mean, cac_std, num_sims)
    cac_samples = np.maximum(cac_samples, 0)  # Ensure non-negative
    
    # Monthly Recurring Revenue (Log-normal)
    mrr_sigma = np.sqrt(np.log(1 + (mrr_std/mrr_mean)**2))
    mrr_mu = np.log(mrr_mean) - 0.5 * mrr_sigma**2
    mrr_samples = np.random.lognormal(mrr_mu, mrr_sigma, num_sims)
    
    # Annual Churn Rate (Beta distribution)
    # Convert mean and std to alpha, beta parameters
    churn_var = churn_rate_std**2
    churn_alpha = churn_rate_mean * ((churn_rate_mean * (1 - churn_rate_mean)) / churn_var - 1)
    churn_beta = (1 - churn_rate_mean) * ((churn_rate_mean * (1 - churn_rate_mean)) / churn_var - 1)
    churn_samples = np.random.beta(max(churn_alpha, 0.1), max(churn_beta, 0.1), num_sims)
    
    # Customer Acquisition Rate (Gamma distribution)
    acq_shape = (acquisition_rate_mean / acquisition_rate_std) ** 2
    acq_scale = acquisition_rate_std ** 2 / acquisition_rate_mean
    acquisition_rate_samples = np.random.gamma(acq_shape, acq_scale, num_sims)
    
    # Customer Acquisition Growth Rate (Normal)
    acquisition_growth_samples = np.random.normal(acquisition_growth_mean, acquisition_growth_std, num_sims)
    
    # Support Cost per Customer (Log-normal)
    support_sigma = np.sqrt(np.log(1 + (support_cost_std/support_cost_mean)**2))
    support_mu = np.log(support_cost_mean) - 0.5 * support_sigma**2
    support_cost_samples = np.random.lognormal(support_mu, support_sigma, num_sims)
    
    # Other costs (Normal)
    fixed_costs_samples = np.random.normal(fixed_costs_mean, fixed_costs_std, num_sims)
    rd_cost_samples = np.random.normal(rd_cost_mean, rd_cost_std, num_sims)
    tech_cost_samples = np.random.normal(tech_cost_mean, tech_cost_std, num_sims)
    
    # Ensure non-negative values
    fixed_costs_samples = np.maximum(fixed_costs_samples, 0)
    rd_cost_samples = np.maximum(rd_cost_samples, 0)
    tech_cost_samples = np.maximum(tech_cost_samples, 0)
    
    return {
        'cac': cac_samples,
        'mrr': mrr_samples,
        'churn_rate': churn_samples,
        'acquisition_rate': acquisition_rate_samples,
        'acquisition_growth': acquisition_growth_samples,
        'support_cost': support_cost_samples,
        'fixed_costs': fixed_costs_samples,
        'rd_cost': rd_cost_samples,
        'tech_cost': tech_cost_samples
    }

def simulate_business(params, months=60):
    """Simulate business for given parameters over specified months (5 years = 60 months)"""
    
    # Initialize arrays
    customers = np.zeros(months + 1)  # +1 for month 0
    monthly_cash_flows = np.zeros(months + 1)
    monthly_revenue = np.zeros(months + 1)
    monthly_ebitda = np.zeros(months + 1)
    new_customers_per_month = np.zeros(months + 1)
    
    # Month 0: Initial R&D cost
    monthly_cash_flows[0] = -params['rd_cost']
    
    # Monthly churn rate (convert annual to monthly)
    monthly_churn_rate = 1 - (1 - params['churn_rate']) ** (1/12)
    
    # Simulate each month
    for month in range(1, months + 1):
        # Calculate new customers with growth
        base_acquisition = params['acquisition_rate'] * (1 + params['acquisition_growth']) ** (month - 1)
        new_customers_per_month[month] = max(0, base_acquisition)
        
        # Update customer count
        # Lose customers due to churn
        customers[month] = customers[month-1] * (1 - monthly_churn_rate)
        # Add new customers
        customers[month] += new_customers_per_month[month]
        
        # Calculate monthly revenue
        revenue = customers[month] * params['mrr']
        monthly_revenue[month] = revenue
        
        # Calculate monthly costs
        customer_acquisition_cost = new_customers_per_month[month] * params['cac']
        variable_costs = customers[month] * params['support_cost']
        fixed_costs = params['fixed_costs'] + params['tech_cost']
        
        # Calculate EBITDA (excluding R&D as it's a one-time upfront cost)
        monthly_ebitda[month] = revenue - customer_acquisition_cost - variable_costs - fixed_costs
        
        # Cash flow is same as EBITDA for ongoing operations
        monthly_cash_flows[month] = monthly_ebitda[month]
    
    # Calculate NPV
    discount_factors = [(1 + discount_rate) ** (-month/12) for month in range(months + 1)]
    npv = np.sum(monthly_cash_flows * discount_factors)
    
    # Calculate ARR at end of period (month 60)
    arr = customers[months] * params['mrr'] * 12
    
    # Calculate Customer Lifetime Value
    # CLV = (Monthly Revenue / Monthly Churn Rate) - Customer Acquisition Cost
    if monthly_churn_rate > 0:
        clv = (params['mrr'] / monthly_churn_rate) - params['cac']
    else:
        clv = np.inf  # If no churn, CLV is infinite
    
    # Calculate annual revenues and EBITDA for each year
    annual_revenues = []
    annual_ebitda = []
    for year in range(5):
        start_month = year * 12 + 1
        end_month = min((year + 1) * 12 + 1, months + 1)
        year_revenue = np.sum(monthly_revenue[start_month:end_month])
        year_ebitda = np.sum(monthly_ebitda[start_month:end_month])
        annual_revenues.append(year_revenue)
        annual_ebitda.append(year_ebitda)
    
    return {
        'npv': npv,
        'arr': arr,
        'clv': clv,
        'final_customers': customers[months],
        'total_customers_acquired': np.sum(new_customers_per_month),
        'monthly_cash_flows': monthly_cash_flows,
        'monthly_revenue': monthly_revenue,
        'customers_over_time': customers,
        'annual_revenues': annual_revenues,
        'annual_ebitda': annual_ebitda
    }

def run_simulation():
    """Function to run the Monte Carlo simulation"""
    with st.spinner("Running Monte Carlo simulation..."):
        # Generate random samples
        samples = generate_random_samples(num_simulations)
        
        # Run simulations
        results = []
        simulation_inputs = []  # Store input parameters for each simulation
        for i in range(num_simulations):
            params = {key: values[i] for key, values in samples.items()}
            result = simulate_business(params)
            results.append(result)
            # Store the input parameters for this simulation
            simulation_inputs.append(params.copy())
        
        # Extract results
        npv_results = [r['npv'] for r in results]
        arr_results = [r['arr'] for r in results]
        clv_results = [r['clv'] for r in results if r['clv'] != np.inf]  # Filter out infinite CLV
        final_customers_results = [r['final_customers'] for r in results]
        annual_revenue_results = [r['annual_revenues'] for r in results]
        annual_ebitda_results = [r['annual_ebitda'] for r in results]
        
        # Store results in session state
        st.session_state.npv_results = npv_results
        st.session_state.arr_results = arr_results
        st.session_state.clv_results = clv_results
        st.session_state.final_customers_results = final_customers_results
        st.session_state.annual_revenue_results = annual_revenue_results
        st.session_state.annual_ebitda_results = annual_ebitda_results
        st.session_state.full_results = results  # Store full results for CSV export
        st.session_state.simulation_inputs = simulation_inputs  # Store input parameters
        st.session_state.simulation_complete = True

# Create a hash of current parameters to detect changes
current_params = {
    'num_simulations': num_simulations,
    'discount_rate': discount_rate,
    'cac_mean': cac_mean,
    'cac_std': cac_std,
    'acquisition_rate_mean': acquisition_rate_mean,
    'acquisition_rate_std': acquisition_rate_std,
    'acquisition_growth_mean': acquisition_growth_mean,
    'acquisition_growth_std': acquisition_growth_std,
    'mrr_mean': mrr_mean,
    'mrr_std': mrr_std,
    'churn_rate_mean': churn_rate_mean,
    'churn_rate_std': churn_rate_std,
    'fixed_costs_mean': fixed_costs_mean,
    'fixed_costs_std': fixed_costs_std,
    'support_cost_mean': support_cost_mean,
    'support_cost_std': support_cost_std,
    'rd_cost_mean': rd_cost_mean,
    'rd_cost_std': rd_cost_std,
    'tech_cost_mean': tech_cost_mean,
    'tech_cost_std': tech_cost_std
}

# Check if parameters have changed or if it's the first run
params_hash = str(hash(str(sorted(current_params.items()))))
if 'previous_params_hash' not in st.session_state:
    st.session_state.previous_params_hash = ""

params_changed = st.session_state.previous_params_hash != params_hash
first_run = not hasattr(st.session_state, 'simulation_complete')

# Auto-run logic
if auto_run and (params_changed or first_run):
    run_simulation()
    st.session_state.previous_params_hash = params_hash

# Manual run button (always available)
if st.sidebar.button("🚀 Run Simulation", type="primary"):
    run_simulation()
    st.session_state.previous_params_hash = params_hash

# Initialize session state variables if they don't exist
if 'simulation_complete' not in st.session_state:
    st.session_state.simulation_complete = False
if 'npv_results' not in st.session_state:
    st.session_state.npv_results = []
if 'arr_results' not in st.session_state:
    st.session_state.arr_results = []
if 'clv_results' not in st.session_state:
    st.session_state.clv_results = []
if 'annual_revenue_results' not in st.session_state:
    st.session_state.annual_revenue_results = []
if 'annual_ebitda_results' not in st.session_state:
    st.session_state.annual_ebitda_results = []
if 'simulation_inputs' not in st.session_state:
    st.session_state.simulation_inputs = []

# Display results if simulation has been run
if st.session_state.simulation_complete and len(st.session_state.npv_results) > 0:
    st.header("📈 Simulation Results")
    
    # Summary statistics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Mean NPV", 
            f"${np.mean(st.session_state.npv_results):,.0f}",
            f"±${np.std(st.session_state.npv_results):,.0f}"
        )
    
    with col2:
        st.metric(
            "Mean ARR", 
            f"${np.mean(st.session_state.arr_results):,.0f}",
            f"±${np.std(st.session_state.arr_results):,.0f}"
        )
    
    with col3:
        if len(st.session_state.simulation_inputs) > 0:
            mrr_values = [params['mrr'] for params in st.session_state.simulation_inputs]
            st.metric(
                "Mean MRR/Customer", 
                f"${np.mean(mrr_values):,.0f}",
                f"±${np.std(mrr_values):,.0f}"
            )
        else:
            st.metric("Mean MRR/Customer", "N/A", "Run simulation")
    
    with col4:
        if len(st.session_state.clv_results) > 0:
            st.metric(
                "Mean CLV", 
                f"${np.mean(st.session_state.clv_results):,.0f}",
                f"±${np.std(st.session_state.clv_results):,.0f}"
            )
        else:
            st.metric("Mean CLV", "N/A", "No finite values")
    
    with col5:
        st.metric(
            "Mean Final Customers", 
            f"{np.mean(st.session_state.final_customers_results):,.0f}",
            f"±{np.std(st.session_state.final_customers_results):,.0f}"
        )
    
    # CSV Download button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📊 Download Simulation Results as CSV", type="secondary", use_container_width=True):
            # Create comprehensive CSV data starting with input assumptions
            csv_data = {
                'Simulation_ID': range(1, len(st.session_state.npv_results) + 1),
            }
            
            # Add input assumption columns (the random values used for each simulation)
            if hasattr(st.session_state, 'simulation_inputs'):
                input_params = st.session_state.simulation_inputs
                csv_data.update({
                    'CAC_Used': [params['cac'] for params in input_params],
                    'MRR_Used': [params['mrr'] for params in input_params],
                    'Churn_Rate_Used': [params['churn_rate'] * 100 for params in input_params],  # Convert to percentage
                    'Acquisition_Rate_Used': [params['acquisition_rate'] for params in input_params],
                    'Acquisition_Growth_Used': [params['acquisition_growth'] * 100 for params in input_params],  # Convert to percentage
                    'Support_Cost_Used': [params['support_cost'] for params in input_params],
                    'Fixed_Costs_Used': [params['fixed_costs'] for params in input_params],
                    'RD_Cost_Used': [params['rd_cost'] for params in input_params],
                    'Tech_Cost_Used': [params['tech_cost'] for params in input_params],
                })
            
            # Add output results
            csv_data.update({
                'NPV': st.session_state.npv_results,
                'ARR': st.session_state.arr_results,
                'Final_Customers': st.session_state.final_customers_results,
            })
            
            # Add CLV - include all values including infinite ones for CSV
            if hasattr(st.session_state, 'full_results'):
                csv_data['CLV'] = [r['clv'] if r['clv'] != np.inf else 'Infinite' for r in st.session_state.full_results]
            
            # Add annual revenue data
            if hasattr(st.session_state, 'annual_revenue_results'):
                for year in range(1, 6):
                    csv_data[f'Revenue_Year_{year}'] = [sim[year-1] for sim in st.session_state.annual_revenue_results]
            
            # Add annual EBITDA data
            if hasattr(st.session_state, 'annual_ebitda_results'):
                for year in range(1, 6):
                    csv_data[f'EBITDA_Year_{year}'] = [sim[year-1] for sim in st.session_state.annual_ebitda_results]
            
            # Create DataFrame and CSV
            df = pd.DataFrame(csv_data)
            csv_string = df.to_csv(index=False)
            
            # Create download button
            st.download_button(
                label="💾 Download CSV File",
                data=csv_string,
                file_name=f"saas_monte_carlo_simulation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Show preview of data
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Showing first 10 rows of {len(df)} total simulations")
            
            # Show data structure explanation
            st.markdown("""
            **CSV Data Structure:**
            - **Input Assumptions**: The random parameter values used for each simulation
            - **Output Results**: NPV, ARR, CLV, Final Customers from each simulation
            - **Annual Breakdowns**: Revenue and EBITDA for each year (Years 1-5)
            
            This allows you to analyze which input combinations lead to specific outcomes!
            """)
    
    st.markdown("---")
    
    # Revenue and EBITDA progression charts (side by side)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Annual Revenue Progression")
        if hasattr(st.session_state, 'annual_revenue_results'):
            # Calculate statistics for annual revenue over time
            annual_revenue_array = np.array(st.session_state.annual_revenue_results)
            years = ['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5']
            
            mean_annual_revenue = np.mean(annual_revenue_array, axis=0)
            std_annual_revenue = np.std(annual_revenue_array, axis=0)
            
            fig_annual = go.Figure()
            
            # Add mean line
            fig_annual.add_trace(go.Scatter(
                x=years,
                y=mean_annual_revenue,
                mode='lines+markers',
                name='Mean Annual Revenue',
                line=dict(color='blue', width=3)
            ))
            
            # Add confidence bands (±1 std dev)
            upper_bound = mean_annual_revenue + std_annual_revenue
            lower_bound = mean_annual_revenue - std_annual_revenue
            
            fig_annual.add_trace(go.Scatter(
                x=years + years[::-1],
                y=np.concatenate([upper_bound, lower_bound[::-1]]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name='±1 Std Dev'
            ))
            
            fig_annual.update_layout(
                title="Annual Revenue with Standard Deviation Bands",
                xaxis_title="Year",
                yaxis_title="Annual Revenue ($)",
                hovermode='x'
            )
            
            st.plotly_chart(fig_annual, use_container_width=True)
        else:
            st.info("Run a simulation to see revenue progression analysis")
    
    with col2:
        st.subheader("💰 EBITDA Progression")
        if hasattr(st.session_state, 'annual_ebitda_results'):
            # Calculate statistics for annual EBITDA over time
            annual_ebitda_array = np.array(st.session_state.annual_ebitda_results)
            years = ['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5']
            
            mean_annual_ebitda = np.mean(annual_ebitda_array, axis=0)
            std_annual_ebitda = np.std(annual_ebitda_array, axis=0)
            
            fig_ebitda = go.Figure()
            
            # Add mean line
            fig_ebitda.add_trace(go.Scatter(
                x=years,
                y=mean_annual_ebitda,
                mode='lines+markers',
                name='Mean Annual EBITDA',
                line=dict(color='green', width=3)
            ))
            
            # Add confidence bands (±1 std dev)
            upper_bound_ebitda = mean_annual_ebitda + std_annual_ebitda
            lower_bound_ebitda = mean_annual_ebitda - std_annual_ebitda
            
            fig_ebitda.add_trace(go.Scatter(
                x=years + years[::-1],
                y=np.concatenate([upper_bound_ebitda, lower_bound_ebitda[::-1]]),
                fill='toself',
                fillcolor='rgba(0,150,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name='±1 Std Dev'
            ))
            
            fig_ebitda.update_layout(
                title="Annual EBITDA with Standard Deviation Bands",
                xaxis_title="Year",
                yaxis_title="Annual EBITDA ($)",
                hovermode='x'
            )
            
            st.plotly_chart(fig_ebitda, use_container_width=True)
        else:
            st.info("Run a simulation to see EBITDA progression analysis")
    
    # NPV distribution on its own row
    st.subheader("NPV Distribution")
    fig_npv = px.histogram(
        x=st.session_state.npv_results,
        nbins=50,
        title="Net Present Value Distribution",
        labels={'x': 'NPV ($)', 'y': 'Frequency'}
    )
    fig_npv.add_vline(
        x=np.mean(st.session_state.npv_results),
        line_dash="dash",
        line_color="red",
        annotation_text="Mean"
    )
    st.plotly_chart(fig_npv, use_container_width=True)
    
    # MRR per customer and ARR distribution histograms (side by side)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💵 Monthly Recurring Revenue per Customer")
        if len(st.session_state.simulation_inputs) > 0:
            mrr_values = [params['mrr'] for params in st.session_state.simulation_inputs]
            fig_mrr = px.histogram(
                x=mrr_values,
                nbins=50,
                title="MRR per Customer Distribution",
                labels={'x': 'MRR per Customer ($)', 'y': 'Frequency'}
            )
            fig_mrr.add_vline(
                x=np.mean(mrr_values),
                line_dash="dash",
                line_color="red",
                annotation_text="Mean"
            )
            st.plotly_chart(fig_mrr, use_container_width=True)
        else:
            st.info("Run a simulation to see MRR distribution")
    
    with col2:
        st.subheader("ARR Distribution")
        fig_arr = px.histogram(
            x=st.session_state.arr_results,
            nbins=50,
            title="Annual Recurring Revenue Distribution",
            labels={'x': 'ARR ($)', 'y': 'Frequency'}
        )
        fig_arr.add_vline(
            x=np.mean(st.session_state.arr_results),
            line_dash="dash",
            line_color="red",
            annotation_text="Mean"
        )
        st.plotly_chart(fig_arr, use_container_width=True)
    
    # CLV and CAC distribution histograms (side by side)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💎 Customer Lifetime Value")
        if hasattr(st.session_state, 'clv_results') and len(st.session_state.clv_results) > 0:
            fig_clv = px.histogram(
                x=st.session_state.clv_results,
                nbins=50,
                title="Customer Lifetime Value Distribution",
                labels={'x': 'CLV ($)', 'y': 'Frequency'}
            )
            fig_clv.add_vline(
                x=np.mean(st.session_state.clv_results),
                line_dash="dash",
                line_color="red",
                annotation_text="Mean"
            )
            st.plotly_chart(fig_clv, use_container_width=True)
        else:
            st.warning("No finite CLV values to display (all customers may have 0% churn)")
    
    with col2:
        st.subheader("💰 Customer Acquisition Cost")
        if hasattr(st.session_state, 'simulation_inputs'):
            cac_values = [params['cac'] for params in st.session_state.simulation_inputs]
            fig_cac = px.histogram(
                x=cac_values,
                nbins=50,
                title="Customer Acquisition Cost Distribution",
                labels={'x': 'CAC ($)', 'y': 'Frequency'}
            )
            fig_cac.add_vline(
                x=np.mean(cac_values),
                line_dash="dash",
                line_color="red",
                annotation_text="Mean"
            )
            st.plotly_chart(fig_cac, use_container_width=True)
        else:
            st.info("Run a simulation to see CAC distribution")
    
    # Percentile analysis
    st.subheader("📊 Percentile Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**NPV Percentiles**")
        npv_percentiles = np.percentile(st.session_state.npv_results, [5, 25, 50, 75, 95])
        npv_df = pd.DataFrame({
            'Percentile': ['5th', '25th', '50th (Median)', '75th', '95th'],
            'NPV ($)': [f"${val:,.0f}" for val in npv_percentiles]
        })
        st.dataframe(npv_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.write("**MRR per Customer Percentiles**")
        if len(st.session_state.simulation_inputs) > 0:
            mrr_values = [params['mrr'] for params in st.session_state.simulation_inputs]
            mrr_percentiles = np.percentile(mrr_values, [5, 25, 50, 75, 95])
            mrr_df = pd.DataFrame({
                'Percentile': ['5th', '25th', '50th (Median)', '75th', '95th'],
                'MRR ($)': [f"${val:,.0f}" for val in mrr_percentiles]
            })
            st.dataframe(mrr_df, use_container_width=True, hide_index=True)
        else:
            st.info("Run a simulation to see MRR percentiles")
    
    with col3:
        st.write("**ARR Percentiles**")
        arr_percentiles = np.percentile(st.session_state.arr_results, [5, 25, 50, 75, 95])
        arr_df = pd.DataFrame({
            'Percentile': ['5th', '25th', '50th (Median)', '75th', '95th'],
            'ARR ($)': [f"${val:,.0f}" for val in arr_percentiles]
        })
        st.dataframe(arr_df, use_container_width=True, hide_index=True)
    
    # Risk analysis
    st.subheader("⚠️ Risk Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        negative_npv_prob = np.mean(np.array(st.session_state.npv_results) < 0) * 100
        st.metric("Probability of Negative NPV", f"{negative_npv_prob:.1f}%")
    
    with col2:
        low_arr_prob = np.mean(np.array(st.session_state.arr_results) < 100000) * 100
        st.metric("Probability of ARR < $100K", f"{low_arr_prob:.1f}%")
    
    with col3:
        high_npv_prob = np.mean(np.array(st.session_state.npv_results) > 1000000) * 100
        st.metric("Probability of NPV > $1M", f"{high_npv_prob:.1f}%")

else:
    if auto_run:
        st.info("🔄 Auto-run is enabled! Change any parameter to see updated results instantly.")
    else:
        st.info("👈 Adjust parameters in the sidebar and click 'Run Simulation' to start the analysis!")
    
    # Show example visualization
    st.subheader("📊 What You'll Get")
    st.markdown("""
    This Monte Carlo simulation will provide:
    
    - **NPV Distribution**: Histogram showing the range of possible Net Present Values
    - **MRR per Customer Distribution**: Histogram showing the range of Monthly Recurring Revenue per customer
    - **ARR Distribution**: Histogram showing the range of possible Annual Recurring Revenue
    - **Annual Revenue Progression**: Chart showing revenue growth over 5 years with confidence bands
    - **EBITDA Progression**: Chart showing EBITDA growth over 5 years with confidence bands
    - **Customer Lifetime Value**: Histogram showing CLV distribution across simulations
    - **Customer Acquisition Cost**: Histogram showing CAC distribution across simulations
    - **CSV Export**: Download all simulation results including input assumptions for external analysis
    - **Summary Statistics**: Mean, standard deviation, and percentiles for all metrics
    - **Risk Analysis**: Probabilities of different outcome scenarios
    
    The simulation accounts for:
    - Uncertainty in all input parameters
    - Customer acquisition growth over time
    - Monthly churn application
    - Proper discounting for NPV calculation
    - Customer lifetime value calculation
    - EBITDA calculation (Revenue - CAC - Support Costs - Fixed Costs)
    - Statistical distributions appropriate for each parameter type
    
    ### 🔄 Auto-Run Feature
    When enabled, the simulation automatically runs whenever you change any parameter, providing instant feedback and making it easy to explore different scenarios in real-time!
    
    ### 📊 Chart Layout
    1. **Summary Metrics**: Key statistics at the top
    2. **Time Series**: Revenue and EBITDA progression over 5 years
    3. **Outcome Distributions**: NPV and ARR histograms
    4. **Parameter vs. Outcome**: CLV and CAC distributions for analysis
    """)

# Model assumptions
with st.expander("📋 Model Assumptions & Methodology"):
    st.markdown("""
    ### Statistical Distributions Used:
    - **Normal Distribution**: Customer Acquisition Cost, Fixed Costs, R&D Cost, Technology Cost, Customer Acquisition Growth Rate
    - **Log-Normal Distribution**: Monthly Recurring Revenue, Support Cost per Customer (ensures positive values)
    - **Beta Distribution**: Annual Churn Rate (bounded between 0% and 100%)
    - **Gamma Distribution**: Customer Acquisition Rate (ensures positive integer-like values)
    
    ### Key Assumptions:
    1. **Time Horizon**: 5 years (60 months)
    2. **Starting Point**: 0 customers at launch
    3. **Churn**: Applied monthly, converted from annual rate
    4. **Growth**: Customer acquisition rate grows monthly by specified growth rate
    5. **Costs**: All costs except R&D are recurring monthly
    6. **Revenue Recognition**: Immediate (no delays)
    7. **Cash Flow Timing**: End-of-month
    
    ### NPV Calculation:
    - Monthly cash flows are discounted to present value
    - Discount rate is applied monthly (annual rate / 12)
    - Initial R&D cost occurs at time 0
    
    ### Customer Lifetime Value (CLV):
    - CLV = (Monthly Revenue per Customer / Monthly Churn Rate) - Customer Acquisition Cost
    - Represents the net present value of the average customer relationship
    - Infinite CLV values (zero churn) are filtered out of histogram display
    
    ### Annual Revenue Progression:
    - Calculated by summing monthly revenues for each 12-month period
    - Shows mean trajectory with ±1 standard deviation bands
    - Demonstrates business growth trajectory and uncertainty
    
    ### EBITDA Calculation:
    - EBITDA = Revenue - Customer Acquisition Cost - Support Costs - Fixed Costs - Technology Costs
    - Excludes initial R&D cost (one-time upfront investment)
    - Represents operational profitability before interest, taxes, depreciation, and amortization
    - Annual EBITDA calculated by summing monthly EBITDA for each 12-month period
    
    ### Auto-Run Functionality:
    - **Real-time Updates**: Simulation runs automatically when parameters change (if enabled)
    - **Parameter Change Detection**: Smart detection of any input modifications
    - **Performance Optimized**: Reduced default simulation count (3,000) for faster auto-runs
    - **Manual Override**: Run simulation button always available for on-demand execution
    - **Instant Feedback**: See results update immediately as you explore scenarios
    
    ### CSV Export Features:
    - **Input Assumptions**: All random parameter values used for each simulation
    - **Output Results**: NPV, ARR, CLV, final customer count for each simulation
    - **Annual Breakdowns**: Revenue and EBITDA for each of the 5 years
    - **Correlation Analysis**: Enables external analysis of which inputs drive specific outcomes
    - **Timestamped Files**: Automatic naming with date/time for version control
    
    ### Risk Considerations:
    - Model assumes independent parameter variations
    - No consideration of market saturation
    - No seasonal effects
    - No competitive responses
    """) 