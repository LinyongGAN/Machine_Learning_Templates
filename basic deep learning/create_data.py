


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    # Create a sample DataFrame
    data = {
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        # 'label': np.random.randint(0, 2, size=100)
    }
    df = pd.DataFrame(data)
    df["label"] = ((df["feature1"] + df["feature2"])>1).astype(int)

    # Save the DataFrame to a CSV file
    df.to_csv('./data/sample_data.csv', index=False)
    print("Sample data created and saved to 'sample_data.csv'")