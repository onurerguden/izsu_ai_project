# izsu_visualizer.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


class IzsuVisualizer:
    def __init__(self, features_path='izsu_features.csv', health_path='izsu_health_factor.csv', save_dir='graphs'):
        self.features_path = features_path
        self.health_path = health_path
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # CSV'leri yükle
        self.df_features = pd.read_csv(features_path)
        self.df_health = pd.read_csv(health_path)
        print(f"[i] Loaded {len(self.df_features)} feature rows and {len(self.df_health)} health records.")

    def plot_correlation_heatmap(self):
        cols = [
            'Alüminyum', 'Arsenik', 'Demir', 'Klorür',
            'pH', 'İletkenlik', 'Oksitlenebilirlik', 'HealthFactor'
        ]
        corr = self.df_features[cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True)
        plt.title('Correlation Heatmap of Key Water Quality Parameters', fontsize=13)
        plt.tight_layout()
        path = os.path.join(self.save_dir, 'correlation_heatmap.pdf')
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        print(f"[✓] Saved correlation heatmap → {path}")

    def plot_hf_trend(self):
        df = self.df_features.copy()
        df['Tarih'] = pd.to_datetime(df['Tarih'], errors='coerce')
        # General average trend
        trend = df.groupby('Tarih')['HealthFactor'].mean().reset_index()

        # Manually assign best and worst locations
        best_loc = 'Altan Aydin İle 4165 Sok.. Kvs Aktepe'
        worst_loc = 'Çanakkale Cad.i̇le 7057 Sok Kavsağıpınarbaşı'

        # Compute trend for best and worst locations
        best_trend = df[df['NoktaAdi'] == best_loc].groupby('Tarih')['HealthFactor'].mean().reset_index()
        worst_trend = df[df['NoktaAdi'] == worst_loc].groupby('Tarih')['HealthFactor'].mean().reset_index()

        plt.figure(figsize=(10, 6))
        plt.plot(trend['Tarih'], trend['HealthFactor'], color='royalblue', marker='o', linewidth=2, label='General Avg')
        plt.plot(best_trend['Tarih'], best_trend['HealthFactor'], color='green', marker='^', linewidth=2, label=f'Best Location: {best_loc}')
        plt.plot(worst_trend['Tarih'], worst_trend['HealthFactor'], color='red', marker='s', linewidth=2, label=f'Worst Location: {worst_loc}')
        plt.title('Temporal Trend of Average Health Factor (HF)', fontsize=13)
        plt.xlabel('Date')
        plt.ylabel('Average Health Factor')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = os.path.join(self.save_dir, 'hf_temporal_trend.pdf')
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        print(f"[i] Best location (manually defined): {best_loc}")
        print(f"[i] Worst location (manually defined): {worst_loc}")
        print(f"[✓] Saved HF trend → {path}")


def main():
    print("=== İzmir Water Quality Visualization ===")
    viz = IzsuVisualizer(
        features_path='data/izsu_features.csv',
        health_path='data/izsu_health_factor.csv',
        save_dir='data/graphs'
    )
    viz.plot_correlation_heatmap()
    viz.plot_hf_trend()
    print("All graphs saved successfully in the 'data/graphs/' folder.")


if __name__ == "__main__":
    main()