import datasets

def inspect_dataset(dataset_path):
    # データセットの読み込み
    dataset = datasets.load_from_disk(dataset_path)

    # データセットの最初のサンプルを表示
    print("データセットの最初のサンプル:")
    print(dataset[0])

if __name__ == "__main__":
    # データセットのパスを指定
    dataset_path = "/home/hata/repo/GIT-training/saved_datasets/train" # 適切なデータセットパスに変更

    # データセットの確認
    inspect_dataset(dataset_path)
