import yfinance as yf
import os

class VIXFetcher:
    def __init__(self, output_dir):
        self.output_dir = output_dir
    
    def fetch_and_save(self):
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Fetch VIX data for last 10 years
        vix = yf.Ticker("^VIX")
        vix_data = vix.history(period="10y")
        
        # Save to CSV
        output_path = os.path.join(self.output_dir, "vix_values.csv")
        vix_data.to_csv(output_path)
        
        print(f"VIX data saved to {output_path}")
        return vix_data

if __name__ == "__main__":
    # Default output directory
    output_dir = "../data"
    
    print("Fetching VIX data...")
    fetcher = VIXFetcher(output_dir)
    data = fetcher.fetch_and_save()
    
    print(f"Successfully fetched {len(data)} rows of VIX data")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Latest VIX value: {data['Close'].iloc[-1]:.2f}")
    print("\nFirst few rows:")
    print(data.head())