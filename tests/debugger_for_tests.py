import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from tests.test_routine import test_routine



if __name__ == "__main__":
    test_routine()
