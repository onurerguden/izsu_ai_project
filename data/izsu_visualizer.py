# izsu_visualizer.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


class IzsuVisualizer:
    def __init__(
        self,
        features_path: str,
        health_path: str,
        save_dir: str = "graphs"
    ):
        self.features_path = features_path
        self.health_path = health_path
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Load datasets
        self.df_features = pd.read_csv(self.features_path)
        self.df_health = pd.read_csv(self.health_path)

        # Parse datetime
        self.df_features["Tarih"] = pd.to_datetime(
            self.df_features["Tarih"], errors="coerce"
        )
        self.df_health["Tarih"] = pd.to_datetime(
            self.df_health["Tarih"], errors="coerce"
        )

        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            "font.size": 11,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11
        })

        print(
            f"[i] Loaded {len(self.df_features)} feature rows "
            f"and {len(self.df_health)} health records."
        )

    # -------------------------------------------------
    # 1. Correlation heatmap (features vs HealthFactor)
    # -------------------------------------------------
    def plot_correlation_heatmap(self):
        cols = [
            "Alüminyum", "Arsenik", "Demir", "Klorür",
            "pH", "İletkenlik", "Oksitlenebilirlik",
            "HealthFactor"
        ]

        corr = self.df_features[cols].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            square=True,
            cbar_kws={"shrink": 0.8}
        )

        path = os.path.join(self.save_dir, "correlation_heatmap.pdf")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[✓] Saved correlation heatmap → {path}")

    # -------------------------------------------------
    # 2. Temporal trend (general + best + worst)
    # -------------------------------------------------
    def plot_hf_trend(self):
        df = self.df_features.copy()

        trend = (
            df.groupby("Tarih", as_index=False)["HealthFactor"]
            .mean()
        )

        best_loc = "Altan Aydin İle 4165 Sok.. Kvs Aktepe"
        worst_loc = "Çanakkale Cad.i̇le 7057 Sok Kavsağıpınarbaşı"

        best_trend = (
            df[df["NoktaAdi"] == best_loc]
            .groupby("Tarih", as_index=False)["HealthFactor"]
            .mean()
        )

        worst_trend = (
            df[df["NoktaAdi"] == worst_loc]
            .groupby("Tarih", as_index=False)["HealthFactor"]
            .mean()
        )

        plt.figure(figsize=(11, 6))
        plt.plot(trend["Tarih"], trend["HealthFactor"],
                 color="royalblue", linewidth=2,
                 label="Overall Mean")

        plt.plot(best_trend["Tarih"], best_trend["HealthFactor"],
                 color="green", linewidth=2,
                 label="Best Location")

        plt.plot(worst_trend["Tarih"], worst_trend["HealthFactor"],
                 color="red", linewidth=2,
                 label="Worst Location")

        plt.xlabel("Date")
        plt.ylabel("Average Health Factor")
        plt.legend(frameon=False)
        plt.grid(alpha=0.3)

        path = os.path.join(self.save_dir, "hf_temporal_trend.pdf")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[✓] Saved HF temporal trend → {path}")

    # -------------------------------------------------
    # 3. Health Factor density
    # -------------------------------------------------
    def plot_hf_density(self):
        plt.figure(figsize=(8, 5))
        sns.kdeplot(
            data=self.df_health,
            x="HealthFactor",
            fill=True,
            alpha=0.6,
            color="#4575b4"
        )

        plt.xlabel("Health Factor")
        plt.ylabel("Density")

        path = os.path.join(self.save_dir, "hf_density.pdf")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[✓] Saved HF density → {path}")

    # -------------------------------------------------
    # 4. Time series (daily + monthly mean)
    # -------------------------------------------------
    def plot_time_series_hf(self):
        plt.figure(figsize=(12, 6))

        sns.lineplot(
            data=self.df_health,
            x="Tarih",
            y="HealthFactor",
            color="gray",
            alpha=0.15,
            label="Daily Observations"
        )

        monthly_avg = (
            self.df_health
            .resample("ME", on="Tarih")["HealthFactor"]
            .mean()
            .reset_index()
        )

        sns.lineplot(
            data=monthly_avg,
            x="Tarih",
            y="HealthFactor",
            color="#1b9e77",
            linewidth=3,
            label="Monthly Mean Trend"
        )

        plt.xlabel("Date")
        plt.ylabel("Health Factor Index")
        plt.legend(frameon=False)

        path = os.path.join(self.save_dir, "hf_time_series.pdf")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[✓] Saved HF time series → {path}")

    # -------------------------------------------------
    # Run all figures
    # -------------------------------------------------
    def run_all(self):
        self.plot_correlation_heatmap()
        self.plot_hf_trend()
        self.plot_hf_density()
        self.plot_time_series_hf()
        print("[✓] All figures generated successfully.")


def main():
    print("=== Izmir Water Quality Visualization ===")
    viz = IzsuVisualizer(
        features_path="data/izsu_features.csv",
        health_path="data/izsu_health_factor.csv",
        save_dir="data/graphs"
    )
    viz.run_all()
    def plot_hf_density(self):
        plt.figure(figsize=(8, 5))
        sns.kdeplot(
            data=self.df,
            x='HealthFactor',
            fill=True,
            color='#4575b4',
            alpha=0.6
        )
        plt.xlabel("Health Factor")
        plt.ylabel("Density")
        plt.tight_layout()

        path = os.path.join(self.save_dir, 'hf_density.png')
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[✓] Saved: {path}")

    def plot_time_series_hf(self):
        plt.figure(figsize=(12, 6))

        # Raw observations
        sns.lineplot(
            data=self.df,
            x='Tarih',
            y='HealthFactor',
            alpha=0.15,
            color='gray',
            label='Daily Observations'
        )

        # Monthly mean trend
        monthly_avg = self.df.set_index('Tarih').resample('M')['HealthFactor'].mean().reset_index()

        sns.lineplot(
            data=monthly_avg,
            x='Tarih',
            y='HealthFactor',
            color='#1b9e77',
            linewidth=3,
            label='Monthly Mean Trend'
        )

        plt.xlabel("Date")
        plt.ylabel("Health Factor Index")
        plt.legend(frameon=False)
        plt.tight_layout()

        path = os.path.join(self.save_dir, 'hf_time_series.png')
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[✓] Saved: {path}")

if __name__ == "__main__":
    main()