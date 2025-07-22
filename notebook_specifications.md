
# Jupyter Notebook Technical Specification: Operational Loss Event Analyzer

This specification details the structure, logical flow, and content of a Jupyter Notebook designed to analyze synthetic operational loss event data, aligning with concepts from the "PRMIA Operational Risk Manager Handbook" [1], [2], [4].

## 1. Notebook Overview

### Learning Goals
The primary learning goals for users interacting with this notebook are:
*   To understand the critical importance of collecting accurate operational loss data for effective risk management, as highlighted in Chapter 5 of the PRMIA Handbook [1].
*   To learn how to identify and analyze the frequency and severity components inherent in operational losses, referencing concepts from Chapter 6 [2].
*   To explore and visualize the statistical properties of loss distributions, such as skewness and heavy tails, which are characteristic of operational risk data.
*   To grasp the fundamental concepts of Expected Loss (EL) and Unexpected Loss (UL) within the context of operational risk, drawing from Chapter 5 [4].
*   To understand how insurance mitigation impacts operational risk, particularly through the lens of excess of loss policies and their effect on aggregated risk [PRMIA Handbook, Chapter 7, pg. 230].

### Expected Outcomes
Upon completing the notebook, users should be able to:
*   Generate a synthetic dataset of operational loss events based on configurable parameters for frequency and severity.
*   Perform essential data validation checks and summarize key statistical properties of the generated loss data.
*   Visualize operational loss trends over time, understand the shape of loss severity distributions, and compare aggregated losses across different risk categories.
*   Interpret core statistical measures (mean, median, standard deviation, skewness, kurtosis) in the context of operational losses.
*   Apply basic filtering and aggregation techniques to operational loss data.
*   Gain practical, hands-on experience that complements theoretical understanding of operational risk data analysis and its challenges.
*   Understand the basic mechanism of how insurance policies (specifically excess of loss) can reduce net operational risk, through direct calculation and visualization.

## 2. Mathematical and Theoretical Foundations

This section will provide the theoretical background necessary for understanding the analysis conducted in the notebook, with clear explanations and LaTeX-formatted formulas.

### 2.1 Operational Loss Data Concepts
Operational loss events are characterized by two primary components:
*   **Frequency:** The number of loss events occurring within a specific period. This is often modeled using a Poisson distribution for a given rate $\lambda$.
*   **Severity:** The financial magnitude of each individual loss event. This is typically modeled using heavy-tailed distributions such as Lognormal or Pareto, reflecting the potential for large, infrequent losses.

The combination of frequency and severity distributions forms the aggregate loss distribution, which is crucial for operational risk capital calculations.

### 2.2 Expected Loss (EL) and Unexpected Loss (UL)
These concepts are fundamental in operational risk management [4].
*   **Expected Loss (EL):** Represents the average loss expected over a given period. It is typically covered by operational provisions and is considered a cost of doing business.
*   **Unexpected Loss (UL):** Represents losses that deviate significantly from the expected average. These losses are not predictable and require capital to absorb.

For illustration of the impact of unexpected loss (UL) due to insurer default in the context of insurance mitigation, as per [PRMIA Handbook, Chapter 7, pg. 229]:
If $PD_A$ is the default probability associated with an A-rating and $L$ is the insured limit, then the expected loss due to insurer default is given by:
$$EL_{default} = PD_A \cdot L$$
The unexpected loss due to insurer default, assumed to be three standard deviations ($\sigma$) from the mean, is approximated as:
$$UL_{default} = 3 \sigma$$
Where $\sigma$ is computed based on a Bernoulli trial of a full limit loss with probability $PD_A$:
$$ \sigma = L \sqrt{PD_A (1 - PD_A)} $$
Thus,
$$UL_{default} = 3 L \sqrt{PD_A (1 - PD_A)}$$
*Note:* The PRMIA Handbook provides an approximation $UL = 3 PD_A (1 - PD_A)L$. This can be interpreted as a simplification or a specific context of calculation where the standard deviation formulation simplifies under certain assumptions, or it might be a typo in the provided text. We will highlight both interpretations.

### 2.3 Modeling Insurance Mitigation
Insurance policies can mitigate operational risk by transferring a portion of potential losses to an insurer. The notebook will focus on a simplified *excess of loss* policy, where the insurer covers losses exceeding a certain deductible, up to a specified limit.

As defined in [PRMIA Handbook, Chapter 7, pg. 230], for an individual loss event $X_i$, an excess of loss policy $P_{d,c}$ with deductible $d$ and cover (limit per loss event) $c$ computes the portion of the risk transferred ($L_{d,c}(X_i)$) as:
$$L_{d,c}(X_i) = \min(\max(X_i - d, 0), c)$$
This formula implies that:
*   If $X_i \le d$, no loss is covered by insurance, so $L_{d,c}(X_i) = 0$.
*   If $d < X_i \le d+c$, the loss covered is $X_i - d$.
*   If $X_i > d+c$, the maximum covered loss is $c$.

For a series of $N$ loss events, the net aggregate risk ($S_{net}$) after the impact of insurance mitigation is given by the total gross aggregate risk minus the total amount recovered from insurance:
$$S_{net} = \sum_{i=1}^{N}X_i - \sum_{i=1}^{N}L_{d,c}(X_i)$$
This shows the reduction in total aggregate loss due to the insurance policy.

### 2.4 Statistical Measures for Loss Distributions
*   **Mean:** The average `Loss Amount`, providing a measure of central tendency.
*   **Median:** The middle value of `Loss Amount`, useful for skewed distributions as it is less affected by outliers.
*   **Standard Deviation:** A measure of the dispersion or spread of `Loss Amount` around the mean.
*   **Skewness:** A measure of the asymmetry of the `Loss Amount` distribution. A positive skewness indicates a "heavy tail" to the right, common in operational loss data, meaning there are a few very large losses.
*   **Kurtosis:** A measure of the "tailedness" of the `Loss Amount` distribution. High kurtosis implies more extreme outliers (heavier tails) than a normal distribution, also typical for operational losses.

## 3. Code Requirements

This section outlines the expected libraries, input/output, algorithms, and visualizations for the notebook.

### 3.1 Expected Libraries
The notebook will utilize common open-source Python libraries available on PyPI.
*   `numpy`: For numerical operations, especially in data generation and statistical calculations.
*   `pandas`: For data manipulation, creating DataFrames, and handling time-series data.
*   `scipy.stats`: For generating random numbers from specified distributions (e.g., Poisson, Lognormal, Pareto) and for statistical functions (skewness, kurtosis).
*   `matplotlib.pyplot`: For basic static plotting capabilities.
*   `seaborn`: For enhanced statistical data visualization, built on Matplotlib.
*   `plotly.graph_objects` and `plotly.express`: For interactive visualizations and user interface elements (sliders, dropdowns), with a static fallback option.
*   `ipywidgets`: For creating interactive controls like sliders and dropdowns within the Jupyter environment.

### 3.2 Input/Output Expectations

#### Input:
*   **User Parameters (via `ipywidgets`):**
    *   **Data Generation:**
        *   `Frequency Rate (Lambda)`: Numeric slider (e.g., 0.1 to 10.0, step 0.1), controlling average events per period.
        *   `Severity Distribution`: Dropdown (e.g., 'Lognormal', 'Pareto').
        *   `Lognormal Mean (mu)`: Numeric slider (e.g., 5.0 to 15.0, step 0.1).
        *   `Lognormal Std Dev (sigma)`: Numeric slider (e.g., 0.5 to 3.0, step 0.1).
        *   `Pareto Scale (b)`: Numeric slider (e.g., 1000 to 10000, step 100).
        *   `Pareto Shape (alpha)`: Numeric slider (e.g., 1.0 to 3.0, step 0.05).
        *   `Number of Periods`: Integer slider (e.g., 12 to 60, for months).
    *   **Analysis/Visualization:**
        *   `Aggregation Period`: Dropdown ('Daily', 'Weekly', 'Monthly', 'Quarterly').
        *   `Filter by Risk Category`: Dropdown (all categories or specific ones).
        *   `Filter by Root Cause`: Dropdown (all causes or specific ones).
        *   `Insurance Deductible (d)`: Numeric slider (e.g., 0 to max loss amount / 5).
        *   `Insurance Cover (c)`: Numeric slider (e.g., 0 to max loss amount / 2).

#### Output:
*   **DataFrames:**
    *   A `pandas.DataFrame` named `loss_events_df` containing the synthetic operational loss data.
*   **Print Statements/Markdown:**
    *   Summary statistics (mean, median, min, max, std dev, skewness, kurtosis) for `Loss Amount`.
    *   Results of data validation checks.
*   **Visualizations:**
    *   Interactive plots (Plotly) for trend, distribution, and aggregated comparison.
    *   Static PNG image files of the plots as fallbacks.

### 3.3 Algorithms or Functions to be Implemented (without code)

#### 3.3.1 Data Generation Module
*   **Function Name:** `generate_synthetic_loss_data`
    *   **Purpose:** To create a synthetic dataset of operational loss events.
    *   **Inputs:** `num_periods`, `freq_rate_lambda`, `severity_dist_type`, `severity_params` (dictionary containing mu, sigma for Lognormal, or b, alpha for Pareto).
    *   **Logic:**
        1.  Generate event dates over `num_periods`.
        2.  For each period, determine the number of events using a Poisson distribution with `freq_rate_lambda`.
        3.  For each event, generate a `Loss Amount` from the specified `severity_dist_type` (Lognormal or Pareto) using `severity_params`.
        4.  Assign a `Loss ID` (unique).
        5.  Randomly assign `Risk Category` (from Basel II Level 2 categories) and `Root Cause`.
        6.  Generate a placeholder `Description`.
        7.  Combine into a `pandas.DataFrame` with columns: `Loss ID`, `Date`, `Risk Category`, `Loss Amount`, `Description`, `Root Cause`.
    *   **Output:** `pandas.DataFrame` of synthetic loss events.

#### 3.3.2 Data Validation Module
*   **Function Name:** `validate_and_summarize_data`
    *   **Purpose:** To perform data quality checks and display summary statistics.
    *   **Inputs:** `loss_events_df` (the generated DataFrame).
    *   **Logic:**
        1.  Check for expected column names: `Loss ID`, `Date`, `Risk Category`, `Loss Amount`, `Description`, `Root Cause`. Raise warning if missing.
        2.  Verify data types: `Loss ID` (numeric/int), `Date` (datetime), `Risk Category` (object/category), `Loss Amount` (numeric/float), `Description` (object), `Root Cause` (object/category).
        3.  Assert `Loss ID` uniqueness (primary key check).
        4.  Assert no missing values in critical fields (`Loss ID`, `Date`, `Loss Amount`). Log if any found.
        5.  Calculate and display summary statistics for `Loss Amount`: mean, median, min, max, standard deviation, skewness, kurtosis.
    *   **Output:** Prints validation results and summary statistics. (Does not return a value, modifies data in place or logs findings).

#### 3.3.3 Insurance Mitigation Calculation
*   **Function Name:** `calculate_insured_loss_and_net_risk`
    *   **Purpose:** To calculate the amount of loss covered by insurance and the resulting net aggregate risk.
    *   **Inputs:** `loss_events_df`, `deductible_d`, `cover_c`.
    *   **Logic:**
        1.  For each `Loss Amount` ($X_i$) in `loss_events_df`, apply the payout function $L_{d,c}(X_i) = \min(\max(X_i - d, 0), c)$ to calculate the `Insured Loss`.
        2.  Add a new column `Insured Loss` to the DataFrame.
        3.  Calculate `Retained Loss` as `Loss Amount` - `Insured Loss`.
        4.  Calculate the total `Gross Aggregate Loss` (sum of `Loss Amount`).
        5.  Calculate the total `Insured Aggregate Loss` (sum of `Insured Loss`).
        6.  Calculate the `Net Aggregate Loss` as `Gross Aggregate Loss` - `Insured Aggregate Loss`.
    *   **Output:** The modified `loss_events_df` with `Insured Loss` and `Retained Loss` columns, and printed summary of aggregate losses.

### 3.4 Visualization Requirements

#### 3.4.1 Trend Plot (Time-Series)
*   **Purpose:** To visualize the total `Loss Amount` over time, identifying trends and high-loss periods.
*   **Type:** Line or Area plot.
*   **Data:** Aggregated `Loss Amount` over user-selected `Aggregation Period` (e.g., monthly, quarterly).
*   **Axes:** X-axis: `Date` (aggregated), Y-axis: `Total Loss`.
*   **Title:** "Operational Loss Event Trend"
*   **Labels:** X: "Aggregation Period", Y: "Total Loss Amount ($)"
*   **Interactivity:** Zooming, hovering for details. Static PNG fallback.

#### 3.4.2 Relationship Plot (Loss Severity Distribution)
*   **Purpose:** To visualize the distribution of `Loss Amount` to understand its shape and heavy tails.
*   **Type:** Histogram overlaid with Kernel Density Estimate (KDE).
*   **Data:** `Loss Amount` column.
*   **Axes:** X-axis: `Loss Amount`, Y-axis: `Frequency` / `Density`.
*   **Title:** "Operational Loss Severity Distribution"
*   **Labels:** X: "Loss Amount ($)", Y: "Frequency / Density"
*   **Additional:** A Q-Q plot comparing `Loss Amount` against a chosen theoretical distribution (e.g., Lognormal) to visually assess fit.
*   **Interactivity:** Zooming, hovering for details. Static PNG fallback.

#### 3.4.3 Aggregated Comparison (Categorical Insights)
*   **Purpose:** To identify key areas of operational risk by comparing losses across `Risk Category` or `Root Cause`.
*   **Type:** Bar Chart or Heatmap.
*   **Data:** Total or average `Loss Amount` grouped by `Risk Category` or `Root Cause`. For heatmap, average loss per category over different time periods (e.g., `Risk Category` vs. `Month`/`Quarter`).
*   **Axes (Bar Chart):** X-axis: `Risk Category` / `Root Cause`, Y-axis: `Total/Average Loss`.
*   **Axes (Heatmap):** X-axis: `Time Period`, Y-axis: `Risk Category`. Color scale: `Average Loss`.
*   **Title:** "Aggregated Loss by Risk Category / Root Cause" or "Average Loss by Category Over Time"
*   **Labels:** Clear labels for axes and color scales.
*   **Interactivity:** Filtering by `Risk Category`/`Root Cause`, zooming. Static PNG fallback.

#### 3.4.4 Insurance Mitigation Visualization
*   **Purpose:** To visually demonstrate the effect of the excess of loss policy on individual loss events.
*   **Type:** Scatter plot or similar visualization showing actual loss, insured portion, and retained portion.
*   **Data:** `Loss Amount`, `Insured Loss`, `Retained Loss` columns.
*   **Axes:** X-axis: `Loss Amount`, Y-axis: `Payout`. A line representing $X_i - L_{d,c}(X_i)$ (retained loss) and another for $L_{d,c}(X_i)$ (transferred loss) against $X_i$.
*   **Title:** "Impact of Insurance Policy on Individual Losses"
*   **Labels:** X: "Gross Loss Amount ($)", Y: "Loss Covered / Retained ($)"
*   **Interactivity:** Sliders for `deductible_d` and `cover_c` to dynamically update the plot, showing how the retained and insured portions change.

### 3.5 Style and Usability
*   All plots should use a color-blind-friendly palette.
*   Font size for titles, labels, and legends should be $\ge 12$ pt.
*   Clear and descriptive titles for all plots.
*   Properly labeled axes with units where appropriate (e.g., "$").
*   Legends provided for multi-series plots.
*   Interactivity (zooming, filtering, tooltips) should be enabled where the environment supports it (e.g., Plotly).
*   For environments without full interactivity, static PNG images of the plots should be generated and saved.

## 4. Additional Notes or Instructions

### 4.1 Assumptions
*   The synthetic data generation will assume a simple Poisson process for frequency and either a Lognormal or Pareto distribution for severity. Real-world data may require more complex distributions or modeling techniques.
*   Basel II Level 2 risk categories will be pre-defined and used for categorical data.
*   The focus is on demonstrating basic operational risk concepts and data analysis, not on building a production-grade risk model.
*   The insurance mitigation model is simplified to an excess of loss policy per event and does not account for aggregate limits, reinstatements, or complex policy conditions as discussed in the PRMIA Handbook, Chapter 7.

### 4.2 Constraints
*   **Execution Time:** The notebook must execute end-to-end on a mid-spec laptop (8 GB RAM) in less than 5 minutes. This implies limiting the size of the synthetic dataset if necessary (e.g., generating data for a few years rather than decades for large organizations).
*   **Libraries:** Only open-source Python libraries from PyPI are permitted.
*   **Code and Narrative:** All major steps in the notebook will include both code comments within the code cells and brief narrative markdown cells immediately preceding the code, explaining "what" is happening and "why" it is being done in the context of operational risk analysis.
*   **Data Sample:** An optional lightweight sample dataset (up to 5 MB) will be provided alongside the notebook, allowing it to run even if a user opts not to generate synthetic data. This sample data should conform to the expected schema (`Loss ID`, `Date`, `Risk Category`, `Loss Amount`, `Description`, `Root Cause`).

### 4.3 Customization Instructions
*   **Data Generation Parameters:** Users can adjust parameters for frequency (Poisson rate) and severity (Lognormal or Pareto parameters) via interactive sliders to observe the impact on loss distributions.
*   **Aggregation Periods:** Users can select different aggregation periods (daily, weekly, monthly, quarterly) for time-series analysis via a dropdown.
*   **Filtering Options:** Users can filter the data by `Risk Category` and `Root Cause` using dropdown menus to focus on specific areas of operational risk.
*   **Insurance Policy Parameters:** Users can adjust the `Deductible (d)` and `Cover (c)` for the simulated insurance policy using sliders to see how different policy terms affect the retained and insured loss amounts.

### 4.4 References
*   [1] Chapter 5: Risk Information & Collecting Loss Data, PRMIA Operational Risk Manager Handbook, Updated November 2015.
*   [2] Chapter 6: Risk Modeling, PRMIA Operational Risk Manager Handbook, Updated November 2015.
*   [3] Loss Models, by Stuart A. Klugman, Harry H. Panjer, Gordon E. Willmot.
*   [4] Expected Loss & Unexpected Loss, PRMIA Operational Risk Manager Handbook, Updated November 2015.
*   [PRMIA Handbook, Chapter 7, pg. 229] Specific reference to the formulas for EL and UL default.
*   [PRMIA Handbook, Chapter 7, pg. 230] Specific reference to the formulas for $L_{d,c}(X_i)$ and $S_{net}$.
