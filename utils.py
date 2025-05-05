from config import*
from train import get_args
def dice_coef_loss(inputs, target, smooth=1e-6):
    """
    Dice Loss: Thước đo sự chồng lấn giữa output và ground truth.
    """
    inputs = torch.sigmoid(inputs)  # Chuyển logits về xác suất
    intersection = (inputs * target).sum()
    union = inputs.sum() + target.sum()
    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice_score  # Dice loss
def bce_dice_loss(inputs, target):
    dice_score = dice_coef_loss(inputs, target)
    bce_loss = nn.BCELoss()
    bce_score = bce_loss(inputs, target)  # yêu cầu đầu vào signmoid rồi => ko dùng code này do model chưa signmoid
    return bce_score + dice_score
def bce_weight_loss(inputs, target, pos_weight = 231.2575):
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE)) #=> yêu cầu model chưa signmoid => dùng code này được
    bce_w_loss = bce(inputs, target)
    return bce_w_loss
def bce_dice_weight_loss(inputs, targets):
    bce_w_loss = bce_weight_loss(inputs, targets) # yêu cầu model chưa signmoid => dùng code này được
    dice = dice_coef_loss(inputs, targets)
    return bce_w_loss + dice
    
def tensor_to_float(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().item()  # Chuyển tensor về CPU và lấy giá trị float
    elif isinstance(value, list):
        return [tensor_to_float(v) for v in value]  # Xử lý danh sách các tensor
    return value  # Nếu không phải tensor, giữ nguyên
def to_numpy(tensor):
    # Move tensor to CPU and convert to NumPy array
    return tensor.cpu().detach().item()
# def dice_coeff(pred, target, smooth=1e-5):
#     pred = torch.sigmoid(pred)  # Chuyển logits về xác suất
#     intersection = torch.sum(pred * target)
#     return (2. * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)
def dice_coeff(pred, target, epsilon=1e-6):
    # print("y_pred_shape1: ", pred.shape)
    # print("target_max:", target.max())
    y_pred = torch.sigmoid(pred)  # Chuyển logits về xác suất
    # print("y_pred_max:", y_pred.max())
    # print("y_pred_shape2: ", y_pred.shape)
    # print("y_target_shape: ", target.shape)
    numerator = 2 * torch.sum(target * y_pred, dim=(1, 2, 3))
    denominator = torch.sum(target + y_pred, dim=(1, 2, 3))
    dice = (numerator + epsilon) / (denominator + epsilon)
    # print("shape_dice: ", dice.shape)
    # return torch.mean(dice)
    return dice

# def iou_core(y_pred, y_true, eps=1e-7):
#     y_pred = torch.sigmoid(y_pred) 
#     y_true_f = y_true.view(-1)  # flatten
#     y_pred_f = y_pred.view(-1)  # flatten

#     intersection = torch.sum(y_true_f * y_pred_f)
#     union = torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection

#     return intersection / (union + eps)  # thêm eps để tránh chia 0
# import torch
# import torch.nn.functional as F
def iou_core(pred, target, epsilon=1e-6):
    pred = torch.sigmoid(pred)  # Chuyển logits về xác suất
    # Tính intersection và union theo từng ảnh
    intersection = torch.sum(pred * target, dim=(1, 2, 3))  # Batch_size x 1
    # print("pred.shape:", pred.shape)
    # print("target.shape:", target.shape)

    union = torch.sum(pred, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3)) - intersection
    iou = (intersection + epsilon) / (union + epsilon)
    return iou  # Giữ nguyên theo batch, mỗi ảnh 1 giá trị

def soft_dice_loss(dice, gamma=0.3):
    """
    Soft Dice Loss dạng Log-Dice, dùng cho segmentation.

    Args:
        y_true: Tensor ground truth, shape (batch_size, H, W)
        y_pred: Tensor prediction, shape (batch_size, H, W)
        epsilon: Giá trị nhỏ để tránh chia cho 0.
        gamma: Hệ số mũ cho log-dice.

    Returns:
        loss: scalar loss value.
    """
    # y_pred = torch.sigmoid(y_pred) 
    # numerator = 2 * torch.sum(y_true * y_pred, dim=(1, 2))
    # denominator = torch.sum(y_true + y_pred, dim=(1, 2))
    # dice = (numerator + epsilon) / (denominator + epsilon)
    # dice = dice_coeff(y_pred, y_true)
    # -------------------------------------------
    # Debug
    # print("dice_in_step:", dice)
    # print("dice_in_step_shape:", dice.shape)
    log_dice = -torch.log(dice)
    # print("log_dice_in_step", log_dice)
    loss = torch.pow(log_dice, gamma)
    # print("loss_in_step", loss)
    # print("loss_in_step_shape:", loss.shape)
    loss_mean = torch.mean(loss)
    # print("loss_mean_in_step", loss_mean)
    # print("loss_mean_in_step_shape:", loss_mean.shape)
    return loss_mean

# def inan():
def loss_func(*kwargs):
    args = get_args()
    if args.loss == "Dice_loss":
        x = dice_coef_loss(*kwargs)
        return x
    elif args.loss == "BCEDice_loss":
        x = bce_dice_loss(*kwargs)
        return x
    elif args.loss == "BCEw_loss":
        x = bce_weight_loss(*kwargs)
        return x
    elif args.loss == "BCEwDice_loss":
        x = bce_dice_weight_loss(*kwargs)
        return x
    elif args.loss == "SoftDice_loss":
        x = soft_dice_loss(*kwargs)
        return x


    
