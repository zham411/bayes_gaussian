# Native Bayes standard.
# Applying Naive Bayes with Gaussian to standard NB.
# Reminder that we are applying Gaussian as a special NB algorithm:
# Specifically to be used when the features have continuous values.
# It assumes all feautures are following a gaussian distribution i.e, normal distribution.


from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def main():

    """
    GaussianNB using sklearn function.
    Iris type dataset was used to for the Mega Donor Model
    If reusing this code, you will need to change the algorithm.
    """

    # Load Iris dataset
    iris = load_iris()

    # Split dataset into train and test data
    X = iris["data"]  # features
    Y = iris["target"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=1
    )

    # Gaussian Naive Bayes
    NB_model = GaussianNB()
    NB_model.fit(x_train, y_train)

    # Display Confusion Matrix
    plot_confusion_matrix(
        NB_model,
        x_test,
        y_test,
        display_labels=iris["target_names"],
        cmap="Blues",
        normalize="true",
    )
    plt.title("Normalized Confusion Matrix - IRIS Dataset")
    plt.show()


if __name__ == "__main__":
    main()
