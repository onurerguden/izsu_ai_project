# izsu_visualizer.py

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class IzsuVisualizer:
    def __init__(
        self,
        features_path: str,
        health_path: str,
        save_dir: str = "graphs",
    ) -> None:
        self.features_path = Path(features_path)
        self.health_path = Path(health_path)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if not self.features_path.exists():
            raise FileNotFoundError(
                f"Features dosyası bulunamadı: {self.features_path}"
            )

        if not self.health_path.exists():
            raise FileNotFoundError(
                f"Health Factor dosyası bulunamadı: {self.health_path}"
            )

        self.df_features = pd.read_csv(
            self.features_path,
            encoding="utf-8-sig",
        )
        self.df_health = pd.read_csv(
            self.health_path,
            encoding="utf-8-sig",
        )

        self._prepare_data()

        plt.rcParams.update(
            {
                "font.size": 11,
                "axes.labelsize": 13,
                "axes.titlesize": 14,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
            }
        )

        print(
            f"[i] Loaded {len(self.df_features)} feature rows "
            f"and {len(self.df_health)} health records."
        )

    def _prepare_data(self) -> None:
        required_feature_columns = {
            "Tarih",
            "NoktaAdi",
            "HealthFactor",
        }
        required_health_columns = {
            "Tarih",
            "NoktaAdi",
            "HealthFactor",
        }

        missing_features = (
            required_feature_columns
            - set(self.df_features.columns)
        )
        missing_health = (
            required_health_columns
            - set(self.df_health.columns)
        )

        if missing_features:
            raise ValueError(
                "Features CSV içinde eksik sütunlar var: "
                + ", ".join(sorted(missing_features))
            )

        if missing_health:
            raise ValueError(
                "Health CSV içinde eksik sütunlar var: "
                + ", ".join(sorted(missing_health))
            )

        self.df_features["Tarih"] = pd.to_datetime(
            self.df_features["Tarih"],
            errors="coerce",
        )
        self.df_health["Tarih"] = pd.to_datetime(
            self.df_health["Tarih"],
            errors="coerce",
        )

        self.df_features["HealthFactor"] = pd.to_numeric(
            self.df_features["HealthFactor"],
            errors="coerce",
        )
        self.df_health["HealthFactor"] = pd.to_numeric(
            self.df_health["HealthFactor"],
            errors="coerce",
        )

        self.df_features = self.df_features.dropna(
            subset=["Tarih", "NoktaAdi", "HealthFactor"]
        ).copy()

        self.df_health = self.df_health.dropna(
            subset=["Tarih", "NoktaAdi", "HealthFactor"]
        ).copy()

        self.df_features = self.df_features.sort_values(
            ["Tarih", "NoktaAdi"]
        ).reset_index(drop=True)

        self.df_health = self.df_health.sort_values(
            ["Tarih", "NoktaAdi"]
        ).reset_index(drop=True)

    def _save_figure(self, filename: str) -> None:
        path = self.save_dir / filename
        plt.tight_layout()
        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"[✓] Saved → {path}")

    # -------------------------------------------------
    # 1. Correlation heatmap
    # -------------------------------------------------
    def plot_correlation_heatmap(self) -> None:
        requested_columns = [
            "Alüminyum",
            "Arsenik",
            "Demir",
            "Klorür",
            "pH",
            "İletkenlik",
            "Oksitlenebilirlik",
            "HealthFactor",
        ]

        available_columns = [
            column
            for column in requested_columns
            if column in self.df_features.columns
        ]

        missing_columns = sorted(
            set(requested_columns) - set(available_columns)
        )

        if missing_columns:
            print(
                "[UYARI] Heatmap için bulunamayan sütunlar: "
                + ", ".join(missing_columns)
            )

        if len(available_columns) < 2:
            print(
                "[UYARI] Correlation heatmap için yeterli sayıda "
                "sayısal sütun bulunamadı."
            )
            return

        correlation_data = self.df_features[
            available_columns
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        corr = correlation_data.corr()

        label_map = {
            "Alüminyum": "Aluminum",
            "Arsenik": "Arsenic",
            "Demir": "Iron",
            "Klorür": "Chloride",
            "pH": "pH",
            "İletkenlik": "Conductivity",
            "Oksitlenebilirlik": "Oxidizability",
            "HealthFactor": "Health Factor",
        }

        corr = corr.rename(
            index=label_map,
            columns=label_map,
        )

        figure_size = max(8, len(corr.columns) * 1.15)
        plt.figure(figsize=(figure_size, figure_size))

        image = plt.imshow(
            corr.values,
            vmin=-1,
            vmax=1,
            aspect="equal",
        )

        plt.colorbar(
            image,
            fraction=0.046,
            pad=0.04,
            label="Pearson Correlation",
        )

        plt.xticks(
            range(len(corr.columns)),
            corr.columns,
            rotation=45,
            ha="right",
        )
        plt.yticks(
            range(len(corr.index)),
            corr.index,
        )

        for row_index in range(len(corr.index)):
            for column_index in range(len(corr.columns)):
                value = corr.iloc[row_index, column_index]

                if pd.notna(value):
                    plt.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                    )

        plt.title(
            "Correlation Matrix of Water Quality Parameters "
            "and Health Factor"
        )

        self._save_figure("correlation_heatmap.pdf")

    # -------------------------------------------------
    # 2. Temporal trend: overall + automatic best/worst
    # -------------------------------------------------
    def plot_hf_trend(self) -> None:
        df = self.df_features.copy()

        overall_trend = (
            df.groupby("Tarih", as_index=False)["HealthFactor"]
            .mean()
            .sort_values("Tarih")
        )

        location_means = (
            df.groupby("NoktaAdi")["HealthFactor"]
            .agg(["mean", "count"])
            .dropna(subset=["mean"])
        )

        # Çok az gözlemi bulunan noktaların yanlış biçimde
        # "best" veya "worst" seçilmesini önlemek için en az
        # üç gözlem şartı uygulanır.
        eligible_locations = location_means[
            location_means["count"] >= 3
        ]

        if eligible_locations.empty:
            eligible_locations = location_means

        if eligible_locations.empty:
            print(
                "[UYARI] Best/worst location trendi için "
                "yeterli nokta verisi bulunamadı."
            )
            return

        best_location = eligible_locations["mean"].idxmax()
        worst_location = eligible_locations["mean"].idxmin()

        best_trend = (
            df[df["NoktaAdi"] == best_location]
            .groupby("Tarih", as_index=False)["HealthFactor"]
            .mean()
            .sort_values("Tarih")
        )

        worst_trend = (
            df[df["NoktaAdi"] == worst_location]
            .groupby("Tarih", as_index=False)["HealthFactor"]
            .mean()
            .sort_values("Tarih")
        )

        plt.figure(figsize=(12, 6))

        plt.plot(
            overall_trend["Tarih"],
            overall_trend["HealthFactor"],
            linewidth=2,
            label="Overall Daily Mean",
        )

        plt.plot(
            best_trend["Tarih"],
            best_trend["HealthFactor"],
            linewidth=1.8,
            label=f"Highest Mean Location: {best_location}",
        )

        plt.plot(
            worst_trend["Tarih"],
            worst_trend["HealthFactor"],
            linewidth=1.8,
            label=f"Lowest Mean Location: {worst_location}",
        )

        plt.xlabel("Date")
        plt.ylabel("Mean Health Factor")
        plt.title(
            "Temporal Trend of Overall, Highest-Mean, "
            "and Lowest-Mean Locations"
        )
        plt.legend(frameon=False)
        plt.grid(alpha=0.3)

        print(
            f"[i] Highest-mean location: {best_location} "
            f"({eligible_locations.loc[best_location, 'mean']:.2f})"
        )
        print(
            f"[i] Lowest-mean location: {worst_location} "
            f"({eligible_locations.loc[worst_location, 'mean']:.2f})"
        )

        self._save_figure("hf_temporal_trend.pdf")

    # -------------------------------------------------
    # 3. Health Factor distribution
    # -------------------------------------------------
    def plot_hf_density(self) -> None:
        values = (
            self.df_health["HealthFactor"]
            .dropna()
            .astype(float)
        )

        if values.empty:
            print(
                "[UYARI] Health Factor distribution için "
                "geçerli veri bulunamadı."
            )
            return

        plt.figure(figsize=(9, 5.5))

        plt.hist(
            values,
            bins=30,
            density=True,
            alpha=0.75,
            edgecolor="black",
        )

        plt.axvline(
            values.mean(),
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {values.mean():.2f}",
        )

        plt.axvline(
            values.median(),
            linestyle=":",
            linewidth=1.5,
            label=f"Median: {values.median():.2f}",
        )

        plt.xlabel("Health Factor")
        plt.ylabel("Density")
        plt.title("Distribution of Health Factor Values")
        plt.legend(frameon=False)
        plt.grid(axis="y", alpha=0.3)

        self._save_figure("hf_density.pdf")

    # -------------------------------------------------
    # 4. Daily and monthly mean time series
    # -------------------------------------------------
    def plot_time_series_hf(self) -> None:
        daily_mean = (
            self.df_health
            .groupby("Tarih", as_index=False)["HealthFactor"]
            .mean()
            .sort_values("Tarih")
        )

        if daily_mean.empty:
            print(
                "[UYARI] Time-series grafiği için "
                "geçerli günlük veri bulunamadı."
            )
            return

        monthly_mean = (
            daily_mean
            .set_index("Tarih")
            .resample("ME")["HealthFactor"]
            .mean()
            .dropna()
            .reset_index()
        )

        plt.figure(figsize=(12, 6))

        plt.plot(
            daily_mean["Tarih"],
            daily_mean["HealthFactor"],
            linewidth=1,
            alpha=0.45,
            label="Daily Mean",
        )

        if not monthly_mean.empty:
            plt.plot(
                monthly_mean["Tarih"],
                monthly_mean["HealthFactor"],
                linewidth=3,
                label="Monthly Mean",
            )

        plt.xlabel("Date")
        plt.ylabel("Health Factor")
        plt.title("Daily and Monthly Mean Health Factor")
        plt.legend(frameon=False)
        plt.grid(alpha=0.3)

        self._save_figure("hf_time_series.pdf")

    # -------------------------------------------------
    # Run all figures
    # -------------------------------------------------
    def run_all(self) -> None:
        self.plot_correlation_heatmap()
        self.plot_hf_trend()
        self.plot_hf_density()
        self.plot_time_series_hf()

        print("[✓] All figures generated successfully.")


def main() -> None:
    print("=== Izmir Water Quality Visualization ===")

    viz = IzsuVisualizer(
        features_path="data/izsu_features.csv",
        health_path="data/izsu_health_factor.csv",
        save_dir="data/graphs",
    )

    viz.run_all()


if __name__ == "__main__":
    main()
