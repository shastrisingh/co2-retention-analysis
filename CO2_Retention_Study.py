import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# --- Helper functions from Cell 1 ---

def week_to_days(week):
    if str(week).lower() == "initial":
        return 0
    elif str(week).lower() in ["24hrs", "24 hours"]:
        return 1
    else:
        try:
            return int(week) * 7
        except:
            return None

def process_sheet(sheet_name, group_by_col, color_map, df_raw):
    df = df_raw.copy()
    df["Week_Days"] = df["Week"].apply(week_to_days)
    relevant_cols = [col for col in df.columns if group_by_col in col]
    df["Avg_CO2"] = df[relevant_cols].mean(axis=1)

    grouped = (
        df.groupby(["Packsize", "Line", "Cap Color", "Week_Days"], as_index=False)
          .agg({"Avg_CO2": "mean"})
          .sort_values(by=["Packsize", "Line", "Cap Color", "Week_Days"])
    )
    grouped["CO2_Diff"] = grouped.groupby(["Packsize", "Line", "Cap Color"])["Avg_CO2"].diff()
    grouped["CO2_Diff"] = -1 * grouped["CO2_Diff"]
    grouped["CO2_Diff"] = grouped["CO2_Diff"].fillna(0)
    
    return grouped

# --- Load Data ---
@st.cache_data
def load_data(file_path):
    capper_df = pd.read_excel(file_path, sheet_name="CapperHead")
    mold_df = pd.read_excel(file_path, sheet_name="MoldNumber")
    return capper_df, mold_df

# --- Main Streamlit app ---

def main():
    st.title("CO₂ Retention Study Dashboard")

    file_path = st.text_input("Excel File Path", 
                             value=r"C:\Users\shastri.doodnath\OneDrive - S.M. Jaleel & Company Limited\Desktop\CO2 Retention Data Entry Sheet.xlsx")
    
    if not file_path:
        st.warning("Please enter the Excel file path.")
        return
    
    capper_df, mold_df = load_data(file_path)
    
    sheet_choice = st.selectbox("Select Sheet to Analyze:", ["CapperHead", "MoldNumber"])
    df_raw = capper_df if sheet_choice == "CapperHead" else mold_df
    group_by_col = "CapperHead" if sheet_choice == "CapperHead" else "MoldNumber"
    
    # Define color map (can customize)
    color_map = {
        "Red": "red",
        "Blue": "blue",
        "Green": "green",
        "Yellow": "gold",
    }
    
    # Process data
    grouped = process_sheet(sheet_choice, group_by_col, color_map, df_raw)

    # Show basic plot (like in cell 1)
    st.subheader(f"CO₂ Retention and Loss Over Time - {sheet_choice}")
    combos = [("355ml", 3), ("355ml", 4), ("500ml", 3)]
    for pack, line in combos:
        df_plot = grouped[(grouped["Packsize"] == pack) & (grouped["Line"] == line)]
        if df_plot.empty:
            continue
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Busta {pack}, Line {line}", fontsize=16, fontweight='bold')

        for cap_color in df_plot["Cap Color"].unique():
            sub = df_plot[df_plot["Cap Color"] == cap_color]
            axs[0].plot(sub["Week_Days"], sub["Avg_CO2"], marker='o', label=f"{cap_color} Retention", color=color_map.get(cap_color, "black"))
            axs[1].plot(sub["Week_Days"], sub["CO2_Diff"], marker='o', label=f"{cap_color} Loss", color=color_map.get(cap_color, "black"))

        axs[0].set_xlabel("Days")
        axs[0].set_ylabel("Average CO₂")
        axs[0].legend()
        axs[0].grid(True)
        
        axs[1].set_xlabel("Days")
        axs[1].set_ylabel("CO₂ Loss")
        axs[1].legend()
        axs[1].grid(True)

        st.pyplot(fig)

        # Print % loss summary
        st.write(f"### % CO₂ loss for Busta {pack} Line {line}")
        for cap_color in df_plot["Cap Color"].unique():
            sub = df_plot[df_plot["Cap Color"] == cap_color]
            initial = sub[sub["Week_Days"] == sub["Week_Days"].min()]["Avg_CO2"].mean()
            latest = sub[sub["Week_Days"] == sub["Week_Days"].max()]["Avg_CO2"].mean()
            if initial and latest:
                pct_loss = ((initial - latest) / initial) * 100
                st.write(f"- **{cap_color}**: {pct_loss:.2f}% loss")
            else:
                st.write(f"- **{cap_color}**: Insufficient data")

    # You can then add buttons or tabs for other analyses (ANOVA, Tukey, Linear Regression ranking, etc.)
    # For example: ANOVA for Cap Color difference (from cell 3)
    
    if st.checkbox("Show ANOVA and Tukey HSD Analysis for Cap Color"):
        st.subheader(f"ANOVA and Tukey HSD - {sheet_choice}")

        combos = [("355ml", 3), ("355ml", 4), ("500ml", 3)]
        df = df_raw.copy()
        df = df.rename(columns={"Cap Color": "Cap_Color"})
        df["Week_Days"] = df["Week"].apply(week_to_days)

        # Calculate Avg_CO2
        relevant_cols = [col for col in df.columns if group_by_col in col]
        df["Avg_CO2"] = df[relevant_cols].mean(axis=1)

        for pack, line in combos:
            subset = df[(df['Packsize'] == pack) & (df['Line'] == line)].dropna(subset=['Avg_CO2', 'Cap_Color'])
            if subset['Cap_Color'].nunique() <= 1:
                st.write(f"Not enough Cap Color groups for Busta {pack}, Line {line}")
                continue
            model = ols('Avg_CO2 ~ C(Cap_Color)', data=subset).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            st.write(f"### Busta {pack}, Line {line}")
            st.dataframe(anova_table)
            p_val = anova_table['PR(>F)'][0]
            if p_val < 0.05:
                tukey = pairwise_tukeyhsd(endog=subset['Avg_CO2'], groups=subset['Cap_Color'], alpha=0.05)
                st.text(tukey.summary())
            else:
                st.write("No significant difference between Cap Colors.")

    # You can continue to add more analyses from your other cells similarly.
    # For example: Linear regression ranking, ANOVA for lines, etc.

if __name__ == "__main__":
    main()
