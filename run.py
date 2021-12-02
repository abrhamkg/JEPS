import torch
from utils.parser import parse_args, load_config
from datasets.loader import construct_loader
from models.jeps_models import AutoencoderJEPS, CompositeJEPS, Generator, EncoderDecoderRNN, MultimodalRNNJEPS, Discriminator
from models.losses import get_loss_func
import torch.optim as optim
import time
from torch.nn.functional import one_hot


def train_GAN(cfg):
    num_epochs = cfg.TRAIN.MAX_EPOCH
    dataloader = construct_loader(cfg)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    nz = cfg.COMPOSITE.GENERATOR_LATENT_DIM
    real_label = 1.
    fake_label = 0.

    netD = Discriminator(cfg).to(device)
    netG = Generator(cfg).to(device)

    criterion = get_loss_func("bce")(reduction=cfg.MODEL.REDUCTION)
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    optimizerD = optim.Adam(netD.parameters(), lr=cfg.MODEL.LEARNING_RATE, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=cfg.MODEL.LEARNING_RATE, betas=(0.5, 0.999))

    print("Starting GAN training:")
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            per_img, goal_img, text, commands, lengths_text, lengths_cmd = data
            # per_img = per_img.to(device)
            goal_img = goal_img.to(device)
            text = text.to(device)
            commands = commands.to(device)

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = goal_img #data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu, text, commands, lengths_text, lengths_cmd).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise, text, commands, lengths_text, lengths_cmd)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach(), text, commands, lengths_text, lengths_cmd).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake, text, commands, lengths_text, lengths_cmd).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            # if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            #     with torch.no_grad():
            #         fake = netG(fixed_noise).detach().cpu()
            #     img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    ckpt_path = f"{cfg.MODEL.CHECKPOINT_DIR}/Generator_{time.time()}.pth"
    print("Saving checkpoint to ", ckpt_path)
    torch.save(netG, ckpt_path)


def train_AEJEPS(cfg):
    num_epochs = cfg.TRAIN.MAX_EPOCH
    dataloader = construct_loader(cfg)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoencoderJEPS(cfg).to(device)
    loss_type = "aejeps_loss"

    criterion = get_loss_func(loss_type)(reduction="none")

    optimizer = optim.Adam(model.parameters(), lr=cfg.MODEL.LEARNING_RATE, betas=(0.5, 0.999))

    print("Started Autoencoder JEPS training")
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for batch_idx, data in enumerate(dataloader, 0):
            per_img, goal_img, text, commands, lengths_text, lengths_cmd = data
            per_img = per_img.to(device)
            goal_img = goal_img.to(device)
            text = text.to(device)
            commands = commands.to(device)

            batch_size = per_img.shape[0]

            train_text, text_target = text[:, :-1], text[:, 1:]
            train_cmd, cmd_target = commands[:, :-1, :], commands[:, 1:, :]

            goal_img_out, text_out, cmd_out = model(per_img, goal_img, train_text, train_cmd, lengths_text, lengths_cmd)

            if loss_type == "aejeps_loss":
                text_out = torch.argmax(text_out, 2)
                cmd_out = torch.argmax(cmd_out, 2)
                cmd_target = torch.argmax(cmd_target, 2)
            loss_img, loss_text, loss_cmd = criterion(goal_img_out, text_out, cmd_out, goal_img, text_target, cmd_target)

            mask_text = torch.zeros(text_out.size()).to(device)
            for i in range(batch_size):
                mask_text[i, :lengths_text[i]] = 1

            # mask_text = mask_text.view(-1).to(device)

            mask_cmd = torch.zeros(cmd_out.size()).to(device)

            for i in range(batch_size):
                mask_cmd[i, :lengths_text[i]] = 1

            # mask_cmd = mask_cmd.view(-1).to(device)

            masked_loss_text = torch.sum(loss_text * mask_text) / torch.sum(mask_text, 1)
            masked_loss_cmd = torch.sum(loss_cmd * mask_cmd) / torch.sum(mask_cmd, 1)

            loss = torch.mean(loss_img) + torch.mean(masked_loss_text) + torch.mean(masked_loss_cmd)# / batch_size
            loss.backward()

            optimizer.step()

            # Output training stats
            if batch_idx % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss: %.4f\t'
                      % (epoch, num_epochs, batch_idx, len(dataloader), loss))

    ckpt_path = f"{cfg.MODEL.CHECKPOINT_DIR}/AEJEPS_{time.time()}.pth"
    print("Saving checkpoint to ", ckpt_path)
    torch.save(model, ckpt_path)


def train_textRNN(cfg):
    num_epochs = cfg.TRAIN.MAX_EPOCH
    dataloader = construct_loader(cfg)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    vocabulary_size = cfg.DATASET.VOCABULARY_SIZE
    num_commands = cfg.DATASET.NUM_COMMANDS
    sos = cfg.DATASET.SOS
    eos = cfg.DATASET.EOS
    embedding_dim = cfg.COMPOSITE.EMBEDDING_DIM
    tg_enc_dim = cfg.COMPOSITE.TG_ENCODER_HIDDEN_DIM
    # tg_dec_dim = cfg.COMPOSITE.TG_DECODER_HIDDEN_DIM
    tg_enc_nlayers = cfg.COMPOSITE.TG_ENCODER_NUM_LAYERS
    tg_bidirectional = cfg.COMPOSITE.TG_BIDIRECTIONAL

    loss_type = "mse"
    model = EncoderDecoderRNN(num_commands, tg_enc_dim, tg_enc_nlayers, vocabulary_size,
                                            embedding_dim, sos, eos, tg_bidirectional, text_sequence=2).to(device)

    # for m in model.parameters():
    #     print(m.requires_grad)

    criterion = get_loss_func(loss_type)(reduction="none")

    optimizer = optim.Adam(model.parameters(), lr=cfg.MODEL.LEARNING_RATE, betas=(0.5, 0.999))

    print("Started Text LSTM training")

    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for batch_idx, data in enumerate(dataloader, 0):
            per_img, goal_img, text, commands, lengths_text, lengths_cmd = data
            per_img = per_img.to(device)
            goal_img = goal_img.to(device)
            text = text.to(device)
            commands = commands.to(device)

            batch_size = per_img.shape[0]

            train_text, text_target = text[:, :-1], text[:, 1:]
            train_cmd, cmd_target = commands[:, :-1, :], commands[:, 1:, :]

            text_out = model(per_img, goal_img, train_cmd, train_text, lengths_text, 'train')
            if loss_type == "mse":
                text_target = one_hot(text_target.long(), num_classes=text_out.shape[2]).float()


            loss_text = criterion(text_out, text_target)

            mask_text = torch.zeros(text_out.size()).to(device)
            for i in range(batch_size):
                mask_text[i, :lengths_text[i]] = 1

            # mask_text = mask_text.view(-1).to(device)

            masked_loss_text = torch.sum(loss_text * mask_text) / torch.sum(mask_text, 1)

            loss = torch.mean(masked_loss_text) # / batch_size

            loss.backward()

            optimizer.step()

            # Output training stats
            if batch_idx % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss: %.4f\t'
                      % (epoch, num_epochs, batch_idx, len(dataloader), loss))

    ckpt_path = f"{cfg.MODEL.CHECKPOINT_DIR}/TextRNN_{time.time()}.pth"
    print("Saving checkpoint to ", ckpt_path)
    torch.save(model, ckpt_path)


def train_cmdRNN(cfg):
    num_epochs = cfg.TRAIN.MAX_EPOCH
    dataloader = construct_loader(cfg)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    vocabulary_size = cfg.DATASET.VOCABULARY_SIZE
    num_commands = cfg.DATASET.NUM_COMMANDS
    sos = cfg.DATASET.SOS
    eos = cfg.DATASET.EOS
    embedding_dim = cfg.COMPOSITE.EMBEDDING_DIM
    cg_enc_dim = cfg.COMPOSITE.CG_ENCODER_HIDDEN_DIM
    # cg_dec_dim = cfg.COMPOSITE.CG_DECODER_HIDDEN_DIM
    cg_enc_nlayers = cfg.COMPOSITE.CG_ENCODER_NUM_LAYERS
    cg_bidirectional = cfg.COMPOSITE.CG_BIDIRECTIONAL

    loss_type = "mse"
    model = EncoderDecoderRNN(vocabulary_size, cg_enc_dim, cg_enc_nlayers, num_commands,
                                               embedding_dim, sos, eos, cg_bidirectional, text_sequence=1).to(device)


    criterion = get_loss_func(loss_type)(reduction=cfg.MODEL.REDUCTION)

    optimizer = optim.Adam(model.parameters(), lr=cfg.MODEL.LEARNING_RATE, betas=(0.5, 0.999))

    print("Started Motor Command LSTM training")

    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for batch_idx, data in enumerate(dataloader, 0):
            per_img, goal_img, text, commands, lengths_text, lengths_cmd = data
            per_img = per_img.to(device)
            goal_img = goal_img.to(device)
            text = text.to(device)
            commands = commands.to(device)

            batch_size = per_img.shape[0]

            train_text, text_target = text[:, :-1], text[:, 1:]
            train_cmd, cmd_target = commands[:, :-1, :], commands[:, 1:, :]

            cmd_out = model(per_img, goal_img, train_text, train_cmd, lengths_text, 'train')


            loss_cmd = criterion(cmd_out, cmd_target)

            mask_cmd = torch.zeros(cmd_out.size()).to(device)
            for i in range(batch_size):
                mask_cmd[i, :lengths_text[i]] = 1

            # mask_cmd = mask_cmd.view(-1).to(device)

            masked_loss_cmd = torch.sum(loss_cmd * mask_cmd) / torch.sum(mask_cmd, 1)

            loss = torch.mean(masked_loss_cmd) # / batch_size
            loss.backward()

            optimizer.step()

            # Output training stats
            if batch_idx % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss: %.4f\t'
                      % (epoch, num_epochs, batch_idx, len(dataloader), loss))

    ckpt_path = f"{cfg.MODEL.CHECKPOINT_DIR}/CommandsRNN_{time.time()}.pth"
    print("Saving checkpoint to ", ckpt_path)
    torch.save(model, ckpt_path)


def main(cfg_path):
    cfg = load_config(cfg_path)

    if cfg.RUN.MODE == "train_gan":
        train_GAN(cfg)

    elif cfg.RUN.MODE == "train_aejeps":
        train_AEJEPS(cfg)

    elif cfg.RUN.MODE == "train_textrnn":
        train_textRNN(cfg)

    elif cfg.RUN.MODE == "train_cmdrnn":
        train_cmdRNN(cfg)
    else:
        raise NotImplementedError(f"{cfg.RUN.MODE} training not implemented")


if __name__ == '__main__':
    args = parse_args()
    main(args.cfg_path)