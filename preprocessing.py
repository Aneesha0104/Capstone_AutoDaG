import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the folder where plots will be saved
output_folder = "dhb1f0sk-13kbsfv92"
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Load the dataset
file_path = "dhb1f0sk-13kbsfv92\\Ecommerce_Behavior.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()
except Exception as e:
    print(f"Error reading the file: {e}")
    exit()

# --- Plotting functions ---


def save_plot(plt, filename):
    """Saves a matplotlib plot to the specified folder."""
    filepath = os.path.join(output_folder, filename)
    try:
        plt.savefig(filepath)
        print(f"Plot saved to: {filepath}")
    except Exception as e:
        print(f"Error saving plot: {e}")


def plot_age_distribution(df):
    """Generates and saves a histogram of customer ages."""
    plt.figure(figsize=(10, 6))
    try:
        plt.hist(df["Age"], bins=20, edgecolor="black")
        plt.title("Age Distribution")
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        save_plot(plt, "age_distribution.png")
        plt.close()
    except KeyError:
        print("Error: 'Age' column not found in the dataset.")
    except Exception as e:
        print(f"Error creating age distribution plot: {e}")


def plot_gender_distribution(df):
    """Generates and saves a bar plot of gender distribution."""
    plt.figure(figsize=(8, 6))
    try:
        gender_counts = df["Gender"].value_counts()
        gender_counts.plot(kind="bar", color=["skyblue", "lightcoral"])
        plt.title("Gender Distribution")
        plt.xlabel("Gender")
        plt.ylabel("Count")
        plt.xticks(rotation=0)  # Keep x-axis labels horizontal
        save_plot(plt, "gender_distribution.png")
        plt.close()
    except KeyError:
        print("Error: 'Gender' column not found.")
    except Exception as e:
        print(f"Error creating gender distribution plot: {e}")


def plot_income_level_distribution(df):
    """Generates and saves a bar plot of income level distribution."""
    plt.figure(figsize=(10, 6))
    try:
        income_counts = df["Income_Level"].value_counts()
        income_counts.plot(kind="bar", color="lightgreen")
        plt.title("Income Level Distribution")
        plt.xlabel("Income Level")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        save_plot(plt, "income_level_distribution.png")
        plt.close()
    except KeyError:
        print("Error: 'Income_Level' column not found.")
    except Exception as e:
        print(f"Error creating income level distribution plot: {e}")


def plot_research_vs_income(df):
    """Generates and saves a box plot of research time vs. income level."""
    plt.figure(figsize=(10, 6))
    try:
        sns.boxplot(
            x="Income_Level", y="Time_Spent_on_Product_Research(hours)", data=df
        )
        plt.title("Time Spent on Product Research vs. Income Level")
        plt.xlabel("Income Level")
        plt.ylabel("Time Spent on Product Research (hours)")
        save_plot(plt, "research_vs_income.png")
        plt.close()
    except KeyError:
        print(
            "Error: 'Income_Level' or 'Time_Spent_on_Product_Research(hours)' column not found."
        )
    except Exception as e:
        print(f"Error creating research vs income plot: {e}")


def plot_research_vs_decision(df):
    """Generates and saves a scatter plot of research time vs. time to decision."""
    plt.figure(figsize=(10, 6))
    try:
        plt.scatter(df["Time_Spent_on_Product_Research(hours)"], df["Time_to_Decision"])
        plt.title("Time Spent on Product Research vs. Time to Decision")
        plt.xlabel("Time Spent on Product Research (hours)")
        plt.ylabel("Time to Decision")
        save_plot(plt, "research_vs_decision.png")
        plt.close()
    except KeyError:
        print(
            "Error: 'Time_Spent_on_Product_Research(hours)' or 'Time_to_Decision' column not found."
        )
    except Exception as e:
        print(f"Error creating research vs decision plot: {e}")


def plot_category_distribution(df):
    """Generates and saves a bar plot of purchase category distribution."""
    plt.figure(figsize=(12, 6))
    try:
        category_counts = df["Purchase_Category"].value_counts()
        category_counts.plot(kind="bar", color="coral")
        plt.title("Purchase Category Distribution")
        plt.xlabel("Purchase Category")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for readability
        save_plot(plt, "category_distribution.png")
        plt.close()
    except KeyError:
        print("Error: 'Purchase_Category' column not found.")
    except Exception as e:
        print(f"Error creating category distribution plot: {e}")


def plot_purchase_amount_by_category(df):
    """Generates and saves a box plot of purchase amount by purchase category."""
    plt.figure(figsize=(12, 6))
    try:
        sns.boxplot(x="Purchase_Category", y="Purchase_Amount", data=df)
        plt.title("Purchase Amount by Purchase Category")
        plt.xlabel("Purchase Category")
        plt.ylabel("Purchase Amount")
        plt.xticks(rotation=45, ha="right")
        save_plot(plt, "purchase_amount_by_category.png")
        plt.close()
    except KeyError:
        print("Error: 'Purchase_Category' or 'Purchase_Amount' column not found.")
    except Exception as e:
        print(f"Error creating purchase amount by category plot: {e}")


def plot_purchase_channel_frequency(df):
    """Generates and saves a bar plot of purchase channel frequency."""
    plt.figure(figsize=(10, 6))
    try:
        channel_counts = df["Purchase_Channel"].value_counts()
        channel_counts.plot(kind="bar", color="skyblue")
        plt.title("Frequency of Purchase by Purchase Channel")
        plt.xlabel("Purchase Channel")
        plt.ylabel("Frequency of Purchase")
        plt.xticks(rotation=0)
        save_plot(plt, "purchase_channel_frequency.png")
        plt.close()
    except KeyError:
        print("Error: 'Purchase_Channel' column not found.")
    except Exception as e:
        print(f"Error creating purchase channel frequency plot: {e}")


def plot_brand_loyalty_vs_rating(df):
    """Generates and saves a scatter plot of brand loyalty vs product rating."""
    plt.figure(figsize=(10, 6))
    try:
        plt.scatter(df["Brand_Loyalty"], df["Product_Rating"])
        plt.title("Brand Loyalty vs. Product Rating")
        plt.xlabel("Brand Loyalty")
        plt.ylabel("Product Rating")
        save_plot(plt, "brand_loyalty_vs_rating.png")
        plt.close()
    except KeyError:
        print("Error: 'Brand_Loyalty' or 'Product_Rating' column not found.")
    except Exception as e:
        print(f"Error creating brand loyalty vs rating plot: {e}")


def plot_satisfaction_vs_return(df):
    """Generates and saves a box plot of customer satisfaction by return rate."""
    plt.figure(figsize=(10, 6))
    try:
        sns.boxplot(x="Return_Rate", y="Customer_Satisfaction", data=df)
        plt.title("Customer Satisfaction by Return Rate")
        plt.xlabel("Return Rate")
        plt.ylabel("Customer Satisfaction")
        save_plot(plt, "satisfaction_vs_return.png")
        plt.close()
    except KeyError:
        print("Error: 'Return_Rate' or 'Customer_Satisfaction' column not found.")
    except Exception as e:
        print(f"Error creating satisfaction vs return plot: {e}")


def plot_discount_vs_purchase_intent(df):
    """Generates and saves a bar plot of discount used vs purchase intent."""
    plt.figure(figsize=(10, 6))
    try:
        cross_tab = pd.crosstab(df["Discount_Used"], df["Purchase_Intent"])
        cross_tab.plot(kind="bar", stacked=False, color=["skyblue", "lightcoral"])
        plt.title("Discount Used vs. Purchase Intent")
        plt.xlabel("Discount Used")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        plt.legend(title="Purchase Intent")
        save_plot(plt, "discount_vs_purchase_intent.png")
        plt.close()
    except KeyError:
        print("Error: 'Discount_Used' or 'Purchase_Intent' column not found.")
    except Exception as e:
        print(f"Error creating discount vs purchase intent plot: {e}")


def plot_payment_method_distribution(df):
    """Generates and saves a bar plot of payment method distribution."""
    plt.figure(figsize=(10, 6))
    try:
        payment_counts = df["Payment_Method"].value_counts()
        payment_counts.plot(kind="bar", color="lightgreen")
        plt.title("Payment Method Distribution")
        plt.xlabel("Payment Method")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        save_plot(plt, "payment_method_distribution.png")
        plt.close()
    except KeyError:
        print("Error: 'Payment_Method' column not found.")
    except Exception as e:
        print(f"Error creating payment method distribution plot: {e}")


# --- Call the plotting functions ---
plot_age_distribution(df)
plot_gender_distribution(df)
plot_income_level_distribution(df)
plot_research_vs_income(df)
plot_research_vs_decision(df)
plot_category_distribution(df)
plot_purchase_amount_by_category(df)
plot_purchase_channel_frequency(df)
plot_brand_loyalty_vs_rating(df)
plot_satisfaction_vs_return(df)
plot_discount_vs_purchase_intent(df)
plot_payment_method_distribution(df)
