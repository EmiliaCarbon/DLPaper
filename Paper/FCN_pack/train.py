import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader
from FCN_pack.load_data import FCNDataset
from FCN_pack.fcn import FCNet

def get_metrics(label_true: np.ndarray, label_pred: np.ndarray):
    """
    :param label_true: has shape (batch_size, n_class, H, W)
    :param label_pred: has the same shape as label_true
    :return: pixel_acc, mean_acc, mIoU of a batch
    """
    batch_size, n_class = label_true.shape[0: 2]
    label_true, label_pred = np.argmax(label_true, axis=1), np.argmax(label_pred, axis=1)   # has shape (batch_size, H, W)

    # conf_mat has shape (n_class, n_class)
    conf_mat = np.array(confusion_matrix(label_true.flatten(), label_pred.flatten(), np.arange(n_class)))
    true_total = np.sum(conf_mat, axis=1)
    pred_total = np.sum(conf_mat, axis=0)
    diag_conf = np.diag(conf_mat)
    pixel_acc = np.sum(diag_conf) / np.sum(true_total)
    mean_acc = np.sum(diag_conf / true_total) / n_class
    mIoU = (diag_conf / (true_total + pred_total - diag_conf)) / n_class

    return pixel_acc, mean_acc, mIoU

def train(train_path, n_class, test_path=None, batch_size=20, learing_rate=10e-4, epoches=150,
          print_info=True, save_path=None):
    train_loader = DataLoader(dataset=FCNDataset(train_path, n_class),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)
    if test_path is not None:
        test_loader = DataLoader(dataset=FCNDataset(test_path, n_class), batch_size=batch_size,
                                 shuffle=True, num_workers=4)
    else:
        test_loader = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FCNet(n_class).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learing_rate, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for epoch in range(epoches):
        model.train()
        step = 1
        for index, (data, label) in enumerate(train_loader):
            data.to(device)
            label.to(device)
            optimizer.zero_grad()
            output = model(data)        # output has shape (batch_size, n_class, W, H)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            if print_info and index % 5 == 0:
                print(f"Epoch {epoch}--{step}/{int(len(train_loader) / 5)}: loss is {loss.item()}")
                step += 1

        if save_path is not None and epoch % 5 == 0:
            torch.save(model.state_dict(), save_path)