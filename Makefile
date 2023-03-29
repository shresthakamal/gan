train:
	python3 -m gan.train

predict:
	python3 -m gan.predict

tensorboard:
	tensorboard --logdir=runs/
