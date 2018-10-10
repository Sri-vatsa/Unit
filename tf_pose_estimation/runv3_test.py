from run_v3 import PoseEstimator

estimator = PoseEstimator()
estimator.configure()

while True:
    estimator.predict()