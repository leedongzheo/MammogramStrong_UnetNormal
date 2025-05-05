import argparse
from dataset import*
# import model.Unet
# import model.Unet
def get_args():
    # Tham số bắt buộc nhập
    parser = argparse.ArgumentParser(description="Train hoặc Pretrain một model AI")
    parser.add_argument("--epoch", type=int, required=True, help="Số epoch để train")
    # parser.add_argument("--model", type=str, required=True, help="Đường dẫn đến model")
    parser.add_argument("--mode", type=str, choices=["train", "pretrain"], required=True, help="Chế độ: train hoặc pretrain")
    parser.add_argument("--data", type=str, required=True, help="Đường dẫn đến dataset đã giải nén")
    # Tham số trường hợp
    parser.add_argument("--checkpoint", type=str, help="Đường dẫn đến file checkpoint (chỉ dùng cho chế độ pretrain)")
    # Tham số mặc định(default)
    parser.add_argument("--saveas", type=str, help="Thư mục lưu checkpoint")
    parser.add_argument("--lr0", type=float, help="learning rate, default = 0.0001")
    parser.add_argument("--batchsize", type=int, help="Batch size, default = 8")

    parser.add_argument("--weight_decay", type=float,  help="weight_decay, default = 1e-6")
    parser.add_argument("--img_size", type=int, nargs=2,  help="Height and width of the image, default = [256, 256]")
    parser.add_argument("--numclass", type=int, help="shape of class, default = 1")
    
    """
    # Với img_size, cách chạy: python script.py --img_size 256 256
    Nếu muốn nhập list dài hơn 3 phần tử, gõ 
    parser.add_argument("--img_size", type=int, nargs='+', default=[256, 256], help="Image dimensions")
    Chạy:
    python script.py --img_size 128 128 3
    """
    parser.add_argument("--loss", type=str, choices=["Dice_loss", "BCEDice_loss", "BCEwDice_loss", "BCEw_loss", "SoftDice_loss"], default="SoftDice_loss", help="Hàm loss sử dụng, default = SoftDice_loss")
    parser.add_argument("--optimizer", type=str, choices=["Adam", "SGD"], default="Adam", help="Optimizer sử dụng, default = Adam")
    return parser.parse_args()
def main():  
    import torch
    from trainer import Trainer
    from model import Unet
    from model import unet_pyramid_cbam_gate
    import optimizer
    from result import export
    global trainer
    SEED=42
    torch.manual_seed(SEED)
    model1 = Unet.Unet(input_channel = 3)
    # model1 = unet_pyramid_cbam_gate.PyramidCbamGateUNet(in_channels=3)
    optimizer1 = optimizer.optimizer(model = model1)
    trainer = Trainer(model = model1, optimizer = optimizer1)
    if args.mode == "train":
        trainer.train(trainLoader, validLoader)
        export(trainer)
    else:
        if not args.checkpoint:
            raise ValueError("Chế độ pretrain yêu cầu checkpoint!")
        trainer.pretrained(train_loader=trainLoader, val_loader=validLoader, checkpoint_path = args.checkpoint)
        # trainer.pretrained(trainLoader,validLoader,args.checkpoint)
        export(trainer)
if __name__ == "__main__":
    args = get_args()
    main()
