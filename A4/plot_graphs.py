import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plotGraph(x, y, xticks=None, yticks=None, xlabel="", ylabel="", title="", filename="graph", ymin=-1, ymax=-1):
	plt.plot(x,y)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	if xticks != None:
		plt.xticks(xticks)
	plt.title(title)
	if ymin != -1:
		plt.ylim(ymin=ymin)
	if ymax != -1:
		plt.ylim(ymax=ymax)
	plt.savefig(filename)
	plt.clf()


def Q2(test_accuracy, training_accuracy, test_loss, training_loss, file_prefix):
	plotGraph(
		[1,2,3,4,5],
		test_accuracy,
		xticks=[1,2,3,4,5],
		xlabel="epoch",
		ylabel="test accuracy",
		title="test accuracy vs the number of epochs",
		filename=file_prefix+"a.png",
		ymin = 0,
		ymax = 1
	)
	plotGraph(
		[1,2,3,4,5],
		training_accuracy,
		xticks=[1,2,3,4,5],
		xlabel="epoch",
		ylabel="training accuracy",
		title="training accuracy vs the number of epochs",
		filename=file_prefix+"b.png",
		ymin = 0,
		ymax = 1
	)
	plotGraph(
		[1,2,3,4,5],
		test_loss,
		xticks=[1,2,3,4,5],
		xlabel="epoch",
		ylabel="test loss",
		title="test loss vs the number of epochs",
		filename=file_prefix+"c.png",
		ymin = 0,
	)
	plotGraph(
		[1,2,3,4,5],
		training_loss,
		xticks=[1,2,3,4,5],
		xlabel="epoch",
		ylabel="training loss",
		title="training loss vs the number of epochs",
		filename=file_prefix+"d.png",
		ymin = 0,
	)

def Q3(rot_acc, blur_acc, file_prefix):
	rotations = [-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, ]
	blurs = [0,1,2,3,4,5,6]
	plotGraph(
		rotations,
		rot_acc,
		xticks=rotations,
		xlabel="degree of rotation",
		ylabel="test accuracy",
		title="test accuracy vs the degree of rotation",
		filename=file_prefix+"a.png",
		ymin = 0,
		ymax = 1
	)
	plotGraph(
		blurs,
		blur_acc,
		xticks=blurs,
		xlabel="blur radius",
		ylabel="test accuracy",
		title="test accuracy vs radius of blur",
		filename=file_prefix+"b.png",
		ymin = 0,
		ymax = 1
	)


Q2(	test_accuracy = [0.7628, 0.8677, 0.9132, 0.9359, 0.9559],
	training_accuracy = [0.5245, 0.8139, 0.8902, 0.9289, 0.9469],
	test_loss = [0.7138, 0.4170, 0.2768, 0.1984, 0.1401],
	training_loss = [1.5100, 0.5612, 0.3544, 0.2342, 0.1745],
	file_prefix = "2"
)

Q3( rot_acc = [0.3604, 0.4587, 0.5556, 0.6602, 0.7549, 0.836, 0.8927, 0.9305, 0.9463, 0.958, 0.9522, 0.9428, 0.9185, 0.8725, 0.8098, 0.7163, 0.6036, 0.4979, 0.3958],
blur_acc = [0.958, 0.9453, 0.7901, 0.4869, 0.2708, 0.1693, 0.0985],
file_prefix = "3")

def Q4():
	testing_acc_regularization = [0.7165, 0.8412, 0.9073, 0.9382, 0.9510]
	training_acc_regularization = [0.4103, 0.7803, 0.8674, 0.9156, 0.9402]
	testing_loss_regularization = [4.5566, 3.8036, 3.3110, 2.9582, 2.7026]
	training_loss_regularization = [5.9487, 4.1626, 3.5729, 3.1517, 2.8447]
	rot_acc_regularization = [0.3691, 0.4677, 0.5829, 0.6854, 0.7811, 0.8556, 0.9052, 0.9348, 0.9456, 0.9514, 0.943, 0.9337, 0.9042, 0.8611, 0.7996, 0.7161, 0.607, 0.5002, 0.4004]
	blur_acc_regularization = [0.9514, 0.933, 0.818, 0.6098, 0.3648, 0.2045, 0.098]

	Q2(	test_accuracy = testing_acc_regularization,
		training_accuracy =training_acc_regularization,
		test_loss = testing_loss_regularization,
		training_loss = training_loss_regularization,
		file_prefix = "4g-2"
	)

	Q3(	rot_acc = rot_acc_regularization,
		blur_acc = blur_acc_regularization,
		file_prefix = "4g-3"
	)

	training_loss_data_aug  = [1.8877, 1.0984, 0.8862, 0.7583, 0.6741]
	training_acc_data_aug = [0.3429, 0.6414, 0.7131, 0.7583, 0.785]
	testing_loss_data_aug = [0.9002, 0.5337, 0.4072, 0.3701, 0.2635]
	testing_acc_data_aug = [0.6994, 0.8203, 0.8699, 0.8756, 0.9186]
	rot_acc_data_aug = [0.6681, 0.7233, 0.7689, 0.8011, 0.8315, 0.8544, 0.8735, 0.8872, 0.8955, 0.9072, 0.8988, 0.8962, 0.8807, 0.8647, 0.8423, 0.8148, 0.7763, 0.725, 0.6706]
	blur_acc_data_aug = [0.9072, 0.8913, 0.778, 0.6279, 0.5198, 0.3961, 0.2192]

	Q2(	test_accuracy = testing_acc_data_aug,
		training_accuracy =training_acc_data_aug,
		test_loss = testing_loss_data_aug,
		training_loss = training_loss_data_aug,
		file_prefix = "4h-2"
	)

	Q3(	rot_acc = rot_acc_data_aug,
		blur_acc = blur_acc_data_aug,
		file_prefix = "4h-3"
	)

Q4()