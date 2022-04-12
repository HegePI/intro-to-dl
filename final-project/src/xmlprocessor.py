import xml_to_csv

if __name__ == "__main__":

    data_processor = xml_to_csv.XmlToCsvWriter(
        "/home/heikki/koulu/intro-to-dl/final-project/data",
        "/home/heikki/koulu/intro-to-dl/final-project/topic_codes.txt",
    )

    codes = data_processor.get_codes()

    train_csv, dev_csv, test_csv = data_processor.write_data_to_csv_files(
        csv_sizes=[7 / 10, 2 / 10, 1 / 10]
    )

    print(train_csv, dev_csv, test_csv)
