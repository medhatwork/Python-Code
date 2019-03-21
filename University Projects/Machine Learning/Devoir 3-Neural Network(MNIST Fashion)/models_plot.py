import matplotlib.pyplot as plt
import numpy as np

def plot(models):
    models = np.array(models)
    
    if models.any().plot_errors == True:
        plot_errors(models)


    if models.any().plot_decision == True:
        plt_dec_boundary(models)
    

def plot_errors(models):
    x_size = int(np.ceil(np.sqrt(len(models))))
    y_size = int(np.ceil(np.sqrt(len(models))))
    fig, axs = plt.subplots(x_size, y_size, constrained_layout=True, figsize=(100,100), squeeze=False)
    fig.suptitle("Errors",  fontweight="bold", fontsize=10)
    i,j = 0, 0

    for model in models:
        if model.plot_errors:
            axs[i][j].set_title(model.plot_errors_title, size=6)
            axs[i][j].plot(model.train_errors/max(model.train_errors), label="Train errors")
            axs[i][j].plot(model.test_errors/max(model.test_errors), label = "Test errors")
            if model.valid_percent > 0:
                 axs[i][j].plot(model.validation_errors/max(model.validation_errors), label = "Cross validation errors")
            axs[i][j].set_xlabel('Epochs',fontsize = 6)
            axs[i][j].set_ylabel('Errors',fontsize = 6)
            if (j+1)%y_size == 0:
                i+=1
                j=0
            else : j+=1
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right')

    plt.show()

    
def plt_dec_boundary(models):
    # Set min and max values and give it some padding
    x_size = int(np.ceil(np.sqrt(len(models))))
    y_size = int(np.ceil(np.sqrt(len(models))))
    fig, axs = plt.subplots(x_size, y_size, constrained_layout=True, figsize=(100,100), squeeze=False)
    fig.suptitle("Decision boundaries", fontsize=10)
    i,j = 0, 0
    for model in models:
        if model.plot_decision:
            X = model.train_inputs
            Y = model.train_labels
            x_min, x_max = X[:, 0].min() - 0.25, X[:, 0].max() + 0.25
            y_min, y_max = X[:, 1].min() - 0.25, X[:, 1].max() + 0.25
            h = 0.01
            # Generate a grid of points with distance h between them
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            # Predict the function value for the whole grid
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            # Plot the contour and training examples
            axs[i][j].contourf(xx, yy, Z, cmap=plt.cm.Spectral)
            axs[i][j].set_title(model.plot_decision_title, size=6)
            axs[i][j].set_xlabel('x1',fontsize = 6)
            axs[i][j].set_ylabel('x2',fontsize = 6)
            axs[i][j].scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)
            if (j+1)%y_size == 0:
                i+=1
                j=0
            else : j+=1
            

    plt.show()

