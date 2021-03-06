import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import numpy as np
from models.cnn_models import resnet50
from einops import rearrange
import math
from torch.nn.functional import one_hot
from utils.parser import AttributeDict
from utils.model_utils import freeze_module


class AutoencoderJEPS(nn.Module):
    """
    This class is an Autoencoder based deep learning implementation of a Joint Episdoic, Procedural, and Semantic Memory.

    Parameters
    ----------
    cfg : utils.parser.AttributeDict
        An AttributeDict containing configuration settings

    Examples
    ---------
    >>> from utils.parser import load_config
    >>> from utils.data_utils import get_bacth
    >>> cfg_file = load_config('configs/default.yaml')
    >>> aejeps = AutoencoderJEPS(cfg_file)
    >>> batch = get_bacth(cfg_file)
    >>> # Using AEJEPS in train mode
    >>> goal_image, lang, cmd = aejeps(*batch, mode='train')
    >>> print(goal_image.shape, lang.shape, cmd.shape)
    >>> # Testing AEJEPS in mode text
    >>> goal_image, lang, cmd = aejeps(*batch, mode='text')
    >>> print(goal_image.shape, lang.shape, cmd.shape)
    >>> # Testing AEJEPS in mode command
    >>> goal_image, lang, cmd = aejeps(*batch, mode='command')
    >>> print(goal_image.shape, lang.shape, cmd.shape)

    """

    def __init__(self, cfg: AttributeDict):
        super().__init__()
        #: The integer representing the start-of-sequence symbol
        self.sos = cfg.DATASET.SOS
        self.eos = cfg.DATASET.EOS
        # Get the parameters for the encoder RNN
        embedding_dim = cfg.AEJEPS.EMBEDDING_DIM
        hidden_dim = cfg.AEJEPS.HIDDEN_DIM
        num_layers_enc = cfg.AEJEPS.NUM_LAYERS_ENCODER
        batch_first = cfg.AEJEPS.BATCH_FIRST
        dropout_rate_enc = cfg.AEJEPS.DROPOUT_ENCODER
        bidirectional_enc = cfg.AEJEPS.BIDIRECTIONAL_ENCODER

        # Get the parameters for the embedding layer
        vocabulary_size = cfg.DATASET.VOCABULARY_SIZE

        # Get the parameters for the motor decoder RNN
        motor_dim = cfg.DATASET.NUM_COMMANDS
        num_layers_motor = cfg.AEJEPS.NUM_LAYERS_MOTOR

        # Get the parameters for the motor decoder RNN
        num_layers_lang = cfg.AEJEPS.NUM_LAYERS_LANG

        # Get the number of motor commands
        num_motor_commands = cfg.DATASET.NUM_COMMANDS

        # Get Image Size
        image_size = cfg.DATASET.IMAGE_SIZE

        # Save the following parameter for use in the forward funciton
        self.num_directions = 2 if bidirectional_enc else 1
        self.num_layers = num_layers_enc
        self.batch_first = batch_first

        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)

        self.feature_extractor_cnn = resnet50(pretrained=True) # Automatically load weights for ImageNet

        encoder_input_dim = embedding_dim + 2 * self.feature_extractor_cnn.fc.in_features + num_motor_commands

        self.encoder = nn.LSTM(encoder_input_dim, hidden_dim, num_layers_enc, batch_first=batch_first,
                               dropout=dropout_rate_enc, bidirectional=bidirectional_enc)

        decoder_hidden_dim = self.num_directions * hidden_dim
        self.motor_decoder = nn.LSTMCell(motor_dim, decoder_hidden_dim, num_layers_motor)

        self.lang_decoder = nn.LSTMCell(embedding_dim, decoder_hidden_dim, num_layers_lang)

        self.hidden_to_conv_in = nn.Linear(decoder_hidden_dim, 1024)
        self.lang_preds = nn.Linear(decoder_hidden_dim, vocabulary_size)
        self.mot_preds = nn.Linear(decoder_hidden_dim, num_motor_commands)

        self.hidden2img = self.__get_transposed_convs(decoder_hidden_dim, image_size)

        # Freeze CNN so it will not be trained
        freeze_module(self.feature_extractor_cnn)

    def __get_transposed_convs(self, decoder_hidden_dim, image_size):
        tconv1 = nn.ConvTranspose2d(1, 4, 3, 2, 3, 0)
        tconv2 = nn.ConvTranspose2d(4, 8, 5, 2, 3, 0)
        tconv3 = nn.ConvTranspose2d(8, 16, 7, 2, 4, 1)
        tconv4 = nn.ConvTranspose2d(16, 3, 11, 1, 7, 0)

        return nn.Sequential(tconv1, tconv2, tconv3, tconv4)

    def forward(self, per_img: torch.Tensor, goal_img: torch.Tensor, text: torch.Tensor, commands: torch.Tensor,
                lengths_text: list, lengths_cmd: list, mode: str='train') -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        This function takes a perceived image, goal image, text, motor command sequence, and the lengths of the text and
        motor command sequence to learn a joint hidden representation from all the modalities. The joint representation
        is then used to reconstruct the three input modalities in train mode. In generation mode two of the input modalities
         are used to generate the third one. In generation mode, the sequence that is being generated needs to be given
         as input. This is required because the encoder operations expect all the inputs they were trained on. Thus,
         to prove correct operation in generation mode a special trivial input with the required size can be given at
         test time. The trivial input may also be used at train time to teach the model to ignore the trivial input.

        Parameters
        ----------
        per_img : torch.Tensor
            A batch of perceived images of shape (batch_size, height, width, 3)
        goal_img : torch.Tensor
            A batch of perceived images of shape (batch_size, height, width, 3)
        text : torch.Tensor
            A batch of padded integer-encoded text input of shape (batch_size, max_sequence_length)
        commands : torch.Tensor
            A batch of padded one-hot encoded motor command sequence input of shape(batch_size, max_sequence_length)
        lengths_text : list
            The lengths of the text sequences in the batch
        lengths_cmd : list
            The lengths of the motor command sequences in the batch
        mode : str
            A string indicating the mode the autoencoder will be used in can be one of 'train', 'image',
                'command', 'text'.<br/>
                'train': Train the autoencoder<br/>
                'image': Use the autoencoder to generate the goal image<br/>
                'command': Use the autoencoder to generate the sequence of motor commands<br/>
                'text': Use the autoencoder to generate the text description

        Returns
        -------
        goal_image_rec : torch.Tensor
            The reconstructed (generated) goal image of shape (batch_size, 3, 224, 224)
        text : torch.Tensor
            The reconstructed (generated) text sequence of shape (batch_size, max_sequence_length, vocabulary_size)
        command : torch.Tensor
            The reconstructed (generated) motor command sequence of shape (batch_size, max_sequence_length, num_motor_commands)
        """

        batch_size, max_len, *_ = text.shape
        num_commands = commands.shape[2]

        _, feats_per = self.feature_extractor_cnn(per_img) # (batch_size, feat_dim)
        _, feats_goal = self.feature_extractor_cnn(goal_img)

        # (batch_size, max_len) -> (batch_size, max_len, embedding_dim)
        text = self.embedding(text.long())

        # For each batch entry determine the length of the longest of the text sequence
        lengths_max = [max(ltext, lcmd) for ltext, lcmd in zip(lengths_text, lengths_cmd)]

        # Batch size x feat_dim -> Batch_size x (max_len x feat_dim) -> Batch_size x max_len x feat_dim
        feats_per = feats_per.repeat((1, max_len)).reshape((batch_size, max_len, -1))
        feats_goal = feats_goal.repeat((1, max_len)).reshape((batch_size, max_len, -1))

        # concatenate the features
        concat_feats = torch.cat((feats_per, feats_goal, text, commands), dim=2)

        packed_input = pack_padded_sequence(concat_feats, lengths_max, enforce_sorted=False, batch_first=self.batch_first)
        output, (hidden, carousel) = self.encoder(packed_input)

        # hidden
        # hidden = hidden.view(self.num_directions, self.num_layers, batch_size, -1)
        # hidden = hidden[:self.num_directions, self.num_layers - 1, :, :]  # Take the last forward direction hidden state for
        hidden = rearrange(hidden, '(d l) b h -> l b (d h)', d=self.num_directions, l=self.num_layers)
        hidden = hidden[self.num_layers - 1,:, :]

        carousel = rearrange(carousel, '(d l) b h -> l b (d h)', d=self.num_directions, l=self.num_layers)
        carousel = carousel[self.num_layers - 1,:, :]

        cmd_h_t, lang_h_t = (hidden, carousel), (hidden, carousel)
        hidden = hidden.unsqueeze(0) # Unsqueeze to match expected input by transposed convolutions

        motor_out = []
        lang_out = []
        device = per_img.device # All tensors must live in the same device

        # Initialize the predictions of the two decoders RNNs at time step t to <sos> value
        prediction_cmd_t = torch.ones(batch_size, 1).to(device).long() * self.sos
        prediction_txt_t = torch.ones(batch_size, 1).to(device).long() * self.sos
        for t in range(max_len):
            # If in train mode use actual inputs, if in generation mode use prediction
            if mode == 'train':
                command = commands[:, t, :]
                char = text[:, t, :]
            elif mode == 'command':
                command = one_hot(prediction_cmd_t.long(), num_classes=num_commands).squeeze(1).float()
                char = text[:, t, :]
            elif mode == 'text':
                command = commands[:, t, :]
                char = self.embedding(prediction_txt_t).squeeze(1)

            # hidden state at time step t for each RNN
            cmd_h_t, cmd_c_t = self.motor_decoder(command, cmd_h_t)
            lang_h_t, lang_c_t = self.lang_decoder(char, lang_h_t)

            cmd_scores = self.mot_preds(cmd_h_t)
            lang_scores = self.lang_preds(lang_h_t)

            cmd_h_t = (cmd_h_t, cmd_c_t)
            lang_h_t = (lang_h_t, lang_c_t)

            motor_out.append(cmd_scores.unsqueeze(1))
            lang_out.append(lang_scores.unsqueeze(1))

            prediction_cmd_t = cmd_scores.argmax(dim=1)
            prediction_txt_t = lang_scores.argmax(dim=1)

        # hidden = hidden[:, :, :self.sqrt_dim ** 2]
        conv_in = self.hidden_to_conv_in(hidden)
        conv_in = rearrange(conv_in, 'l b (h1 h2) -> b l h1 h2', h1=32, h2=32)
        goal_image_rec = self.hidden2img(conv_in)
        return goal_image_rec, torch.cat(lang_out, 1), torch.cat(motor_out, 1)


class CompositeJEPS(nn.Module):
    """
    This class is a deep learning implementation of a Joint Episdoic, Procedural, and Semantic Memory with three
    separate models. The three models are an image generator that generates the goal image based on the text input and
    motor command sequence input, a text generator that generates text from motor command sequence and image features, and
    a motor command sequence generator that generates motor command sequence from image features and text.

    Parameters
    ----------
    cfg : utils.parser.AttributeDict
        An AttributeDict containing configuration settings

    Examples
    ---------
    >>> from utils.parser import load_config
    >>> from utils.data_utils import get_bacth
    >>> cfg_file = load_config('configs/default.yaml')
    >>> compjeps = CompositeJEPS(cfg_file)
    >>> per_image, goal_image, text, commands, lengths_text, lengths_cmd = get_bacth(cfg_file)

    >>> outputs = compjeps(per_image, lengths_text, goal_img=goal_image, text=text)
    >>> print([i.shape for i in outputs if i is not None])

    >>> outputs = compjeps(per_image, lengths_cmd, goal_img=goal_image, commands=commands)
    >>> print([i.shape for i in outputs if i is not None])

    >>> outputs = compjeps(per_image, lengths_text, commands=commands, text=text, lengths_cmd=lengths_cmd, verbose=True)
    >>> print([i.shape for i in outputs if i is not None])

    """

    def __init__(self, cfg : AttributeDict):
        super().__init__()

        # Get some settings form the configuration
        vocabulary_size = cfg.DATASET.VOCABULARY_SIZE
        num_commands = cfg.DATASET.NUM_COMMANDS
        sos = cfg.DATASET.SOS
        eos = cfg.DATASET.EOS

        embedding_dim = cfg.COMPOSITE.EMBEDDING_DIM
        tg_enc_dim = cfg.COMPOSITE.TG_ENCODER_HIDDEN_DIM
        # tg_dec_dim = cfg.COMPOSITE.TG_DECODER_HIDDEN_DIM
        tg_enc_nlayers = cfg.COMPOSITE.TG_ENCODER_NUM_LAYERS
        tg_bidirectional = cfg.COMPOSITE.TG_BIDIRECTIONAL
        cg_enc_dim = cfg.COMPOSITE.CG_ENCODER_HIDDEN_DIM
        # cg_dec_dim = cfg.COMPOSITE.CG_DECODER_HIDDEN_DIM
        cg_enc_nlayers = cfg.COMPOSITE.CG_ENCODER_NUM_LAYERS
        cg_bidirectional = cfg.COMPOSITE.CG_BIDIRECTIONAL

        # Instantiate the three models
        self.image_generator = Generator(cfg)
        self.text_generator = EncoderDecoderRNN(num_commands, tg_enc_dim, tg_enc_nlayers, vocabulary_size,
                                                   embedding_dim, sos, eos, tg_bidirectional, text_sequence=2)

        self.command_generator = EncoderDecoderRNN(vocabulary_size, cg_enc_dim, cg_enc_nlayers, num_commands,
                                                   embedding_dim, sos, eos, cg_bidirectional, text_sequence=1)

    def forward(self, per_img: torch.Tensor, lengths: list, goal_img: torch.Tensor=None, text: torch.Tensor=None,
                commands: torch.Tensor=None, verbose: bool=False, lengths_cmd: list=None):
        """
        This function takes a perceived image and two of the goal image, text, and motor command sequence to generate the
        missing modality. This model is intended to be used in generation mode only. The three models are expected to be
        trained separately. The learned weights can then be loaded for generation.

        The function returns a tuple of three values when generating any of the three modalities. The tuple entries for
        the modality being generated will be the output of the model that generated it while the other two will be None.

        Parameters
        ----------
        per_img : torch.Tensor
            A batch of perceived images of shape (batch_size, height, width, 3)
        goal_img : torch.Tensor
            A batch of perceived images of shape (batch_size, height, width, 3)

        text : torch.Tensor
            A batch of padded integer-encoded text input of shape (batch_size, max_sequence_length)
        commands : torch.Tensor
            A batch of padded one-hot encoded motor command sequence input of shape(batch_size, max_sequence_length)
        lengths_text : list
            The lengths of the text sequences in the batch
        lengths_cmd : list
            The lengths of the motor command sequences in the batch
        verbose : bool
            Enable verbose output.
        Returns
        -------
        goal_image_rec : torch.Tensor or None
            The reconstructed (generated) goal image of shape (batch_size, 3, 224, 224)
        text : torch.Tensor or None
            The reconstructed (generated) text sequence of shape (batch_size, max_sequence_length, vocabulary_size)
        command : torch.Tensor or None
            The reconstructed (generated) motor command sequence of shape (batch_size, max_sequence_length, num_motor_commands)
        """
        # Check that only two inputs are given
        param_count = (text is not None) + (goal_img is not None) + (commands is not None)
        if param_count != 2:
            raise TypeError(f"Expected only 2 of (goal_image, text, commands) to be given but given {param_count} of "
                            f"them")

        if goal_img is None:
            if verbose:
                print("Running CompositeJEPS in image generation mode")
            batch_size = text.shape[0]
            noise_input = torch.randn(batch_size, 256, 1, 1)
            return self.image_generator(noise_input, text, commands, lengths, lengths_cmd), None

        if text is None:
            print("Running CompositeJEPS in text generation mode")
            return None, self.text_generator(per_img, goal_img, commands, text, lengths, mode='test'), None

        if commands is None:
            print("Running CompositeJEPS in command generation mode")
            return None, None, self.command_generator(per_img, goal_img, text, commands, lengths, mode='test')


class EncoderDecoderRNN(nn.Module):
    """
    This class is a generic encoder-decoder architecture implementation with an LSTM encoder and an LSTM decoder. This
    class can be used to generate one sequence from another sequence. This class expects either the encoded sequence
    or the decoded sequence to be text the other sequence is expected to be a one-hot encoded motor command sequence.
    The sizes of the submodules of this module and the operations in the methods of this class depend on which of the
    two sequences is the text.

    Parameters
    ----------
    seq1_elem_count : int
        The number unique sequence elements present in the encoded sequence
    encoder_hidden : int
        The hidden dimension of the encoder.
    encoder_nlayers : int
        The number of layers in the encoder
    seq2_elem_count : int
        The number unique sequence elements present in the sequence to be generated
    embedding_dim : int
        The size of the embedding dimension
    sos : int
        The integer representing start-of-sequence symbol
    eos : int
        The integer representing end-of-sequence symbol
    bidirectional_enc : bool
        Whether to use a bidirectional encoder or unidirectional
    batch_first : bool
        If True the first dimension will be treated as the batch dimension. False not supported now
    text_sequence : int
        If text_sequence is 1, the first sequence will be treated as text. If text_sequence is 2, the second
        sequence will be treated as text. It should be 1 when generating motor commands and 2 when generating text.
    """

    def __init__(self, seq1_elem_count, encoder_hidden, encoder_nlayers, seq2_elem_count,
                 embedding_dim, sos, eos, bidirectional_enc, batch_first=True, text_sequence=1):
        super().__init__()
        self.sos = sos
        self.eos = eos

        self.feature_extractor_cnn = resnet50(pretrained=True) # Automatically load weights for ImageNet
        if text_sequence == 1:
            self.embedding = nn.Embedding(seq1_elem_count, embedding_dim)
            encoder_input_dim = embedding_dim + 2 * self.feature_extractor_cnn.fc.in_features
            num_outputs = seq2_elem_count
            decoder_input_dim = seq2_elem_count
        elif text_sequence == 2:
            self.embedding = nn.Embedding(seq2_elem_count, embedding_dim)
            encoder_input_dim = seq1_elem_count + 2 * self.feature_extractor_cnn.fc.in_features
            num_outputs = seq2_elem_count
            decoder_input_dim = embedding_dim

        self.batch_first = batch_first
        self.seq1_elem_count = seq1_elem_count
        self.seq2_elem_count = seq2_elem_count
        self.text_sequence = text_sequence # Save text sequence for use in forward
        self.num_directions = 2 if bidirectional_enc else 1

        self.encoder = nn.LSTM(encoder_input_dim, encoder_hidden, encoder_nlayers, batch_first=batch_first)

        decoder_hidden_dim = self.num_directions * encoder_hidden

        self.decoder = nn.LSTMCell(decoder_input_dim, decoder_hidden_dim)

        self.seq2_out_fc = nn.Linear(decoder_hidden_dim, num_outputs)

        # Freeze CNN so it will not be trained
        freeze_module(self.feature_extractor_cnn)

    def forward(self, per_img: torch.Tensor, goal_img: torch.Tensor, sequence1: torch.Tensor, sequence2: torch.Tensor,
                lengths1: list, mode: str='train'):
        """

        Parameters
        ----------
        per_img : torch.Tensor
            A batch of perceived images of shape (batch_size, height, width, 3)
        goal_img : torch.Tensor
            A batch of perceived images of shape (batch_size, height, width, 3)
        sequence1 : torch.Tensor
            A batch of padded integer-encoded text input of shape (batch_size, max_sequence_length) or a batch of
            padded one-hot encoded motor command sequence input of shape(batch_size, max_sequence_length)
        sequence2 : torch.Tensor
            A batch of padded integer-encoded text input of shape (batch_size, max_sequence_length) or a batch of
            padded one-hot encoded motor command sequence input of shape(batch_size, max_sequence_length)
        lengths1 : list
            The lengths of sequences in the first sequence (the encoded sequence) in the batch

        mode : str
            The mode to use this model in. mode can be one of 'train', or 'test'
            In 'train' mode the first sequence, the images are input to the encoder while the second sequence is used
            as input to the decoder.
            In 'test' mode the first sequence, the images are input to the encoder while the second sequence is not used
            as input to the decoder. The decoder's input will be the output of the decoder at the previous time step.


        Returns
        -------

        """
        batch_size, max_len1, *_ = sequence1.shape
        if mode == 'train':
            _, max_len2, *_ = sequence2.shape
        else:
            max_len2 = 100

        num_seq2 = self.seq2_elem_count

        _, feats_per = self.feature_extractor_cnn(per_img) # (batch_size, feat_dim)
        _, feats_goal = self.feature_extractor_cnn(goal_img)

        if self.text_sequence == 1:
            sequence1 = self.embedding(sequence1.long())
        elif self.text_sequence == 2 and mode == 'train':
            sequence2 = self.embedding(sequence2.long())
            # sequence1 = one_hot(sequence1.long(), num_classes=self.seq1_size)

        # For each batch entry determine the length of the longest of the text sequence
        # Batch size x feat_dim -> Batch_size x (max_len x feat_dim) -> Batch_size x max_len x feat_dim
        feats_per = feats_per.repeat((1, max_len1)).reshape((batch_size, max_len1, -1))
        feats_goal = feats_goal.repeat((1, max_len1)).reshape((batch_size, max_len1, -1))

        # concatenate the features
        concat_feats = torch.cat((feats_per, feats_goal, sequence1), dim=2)

        packed_input = pack_padded_sequence(concat_feats, lengths1, enforce_sorted=False, batch_first=self.batch_first)
        output, hidden = self.encoder(packed_input)

        seq2_out = []
        device = per_img.device # All tensors must live in the same device
        seq2_h_t = [h[-1, :, :] for h in hidden]

        # Initialize the predictions of the two decoders RNNs at time step t to <sos> value
        prediction_seq2_t = torch.ones(batch_size, 1).to(device).long() * self.sos
        for t in range(max_len2):
            # If in train mode use actual inputs, if in generation mode use prediction
            if mode == 'train':
                seq2_t = sequence2[:, t, :]
            else:
                if self.text_sequence == 2:
                    seq2_t = self.embedding(prediction_seq2_t).squeeze(1)
                elif self.text_sequence == 1:
                    seq2_t = one_hot(prediction_seq2_t, num_classes=num_seq2).squeeze(1).float()

            # hidden state at time step t for the decoder
            seq2_h_t = self.decoder(seq2_t, seq2_h_t)

            seq2_pred = self.seq2_out_fc(seq2_h_t[0])

            seq2_out.append(seq2_pred.unsqueeze(1))

            prediction_seq2_t = seq2_pred.argmax(dim=1)

        return torch.cat(seq2_out, 1)


class Generator(nn.Module):
    """
    This model generates images from a learned image distribution conditioned on a text and motor command sequence input.
    It learns the distribution by attempting to fool the models.jeps_models.Discriminator which attempts to discern
    between real images and images generated by this class.

    Parameters
    ----------
    cfg : utils.parser.AttributeDict
        An AttributeDict containing configuration settings

    Examples
    ---------
    >>> from utils.parser import load_config
    >>> from utils.data_utils import get_bacth
    >>> cfg_file = load_config('configs/default.yaml')

    >>> generator = Generator(cfg_file)
    >>> out = generator(torch.randn(4, 256, 1, 1), text, commands, lengths_text, lengths_cmd)
    >>> print(out.shape)

    """

    def __init__(self, cfg):
        super(Generator, self).__init__()
        nc = 3

        # Size of z latent vector (i.e. size of generator input)
        nz = cfg.COMPOSITE.GENERATOR_LATENT_DIM + cfg.COMPOSITE.TEXT_ENCODING_DIM + cfg.COMPOSITE.COMMAND_ENCODING_DIM

        # Size of feature maps in generator
        ngf = cfg.COMPOSITE.GENERATOR_NUM_CHANNELS

        self.embedding = nn.Embedding(cfg.DATASET.VOCABULARY_SIZE, cfg.COMPOSITE.EMBEDDING_DIM)
        self.fc_text = nn.Linear(cfg.COMPOSITE.EMBEDDING_DIM, cfg.COMPOSITE.TEXT_ENCODING_DIM)
        self.fc_cmd = nn.Linear(cfg.DATASET.NUM_COMMANDS, cfg.COMPOSITE.COMMAND_ENCODING_DIM)
        self.num_commands = cfg.DATASET.NUM_COMMANDS

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf * 2, ngf, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, ngf, 1, 2, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 112 x 112
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 224 x 224

        )

    def forward(self, input: torch.Tensor, text: torch.Tensor, commands: torch.Tensor, lengths_text: list,
                lengths_cmd: list):
        """

        Parameters
        ----------
        input : torch.Tensor
            A noise prior input of shape (batch_size, 1, 1, 1)
        text : torch.Tensor
            A batch of padded integer-encoded text input of shape (batch_size, max_sequence_length)
        commands : torch.Tensor
            A batch of padded one-hot encoded motor command sequence input of shape (batch_size, max_sequence_length)
        lengths_text : list
            The lengths of the text sequences in the batch
        lengths_cmd : list
            The lengths of the motor command sequences in the batch

        Returns
        -------
        A batch of generated images with shape (batch_size, 3, 224, 224)

        """
        batch_size = input.shape[0]

        text = self.embedding(text.long())

        text_encodings = []
        cmd_encodings = []
        for b in range(batch_size):
            encoding = self.fc_text(text[b, :lengths_text[b], :])
            text_encodings.append(encoding)

        for b in range(batch_size):
            encoding = self.fc_cmd(commands[b, :lengths_cmd[b], :])
            cmd_encodings.append(encoding)


        text_encodings = pad_sequence(text_encodings, batch_first=True)
        cmd_encodings = pad_sequence(cmd_encodings, batch_first=True)
        text_encodings = text_encodings.sum(dim=1)
        cmd_encodings = cmd_encodings.sum(dim=1)
        text_encodings = text_encodings.unsqueeze(2).unsqueeze(2)
        cmd_encodings = cmd_encodings.unsqueeze(2).unsqueeze(2)

        input = torch.cat([input, cmd_encodings, text_encodings], 1)
        return self.main(input)


class Discriminator(nn.Module):
    """
    This attempts to classify whether an input image as a real image taken from a dataset or an image generated by the
    models.jeps_models.Generator class. The Discriminator class can also be trained to identify mismatched image, and text
    or motor command sequence as the features from the two sequences are injected into the intermediate layers.

    Parameters
    ----------
    cfg : utils.parser.AttributeDict
        An AttributeDict containing configuration settings

    Examples
    ---------
    >>> from utils.parser import load_config
    >>> from utils.data_utils import get_bacth
    >>> cfg_file = load_config('configs/default.yaml')

    >>> discriminator = Discriminator(cfg_file)

    >>> out = discriminator(out, text, commands, lengths_text, lengths_cmd)
    >>> print("Discriminator", out.shape)

    """
    def __init__(self, cfg):
        super(Discriminator, self).__init__()
        nc = 3

        ndf = cfg.COMPOSITE.DISCRIMINATOR_NUM_CHANNELS

        self.embedding = nn.Embedding(cfg.DATASET.VOCABULARY_SIZE, cfg.COMPOSITE.EMBEDDING_DIM)
        self.fc_text = nn.Linear(cfg.COMPOSITE.EMBEDDING_DIM, cfg.COMPOSITE.TEXT_ENCODING_DIM)
        self.fc_cmd = nn.Linear(cfg.DATASET.NUM_COMMANDS, cfg.COMPOSITE.COMMAND_ENCODING_DIM)

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 1, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 8, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final = nn.Sequential(
            nn.Conv2d(24, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, text, commands, lengths_text, lengths_cmd):
        batch_size = input.shape[0]

        conv_out = self.main(input)

        text = self.embedding(text.long())

        text_encodings = []
        cmd_encodings = []
        for b in range(batch_size):
            encoding = self.fc_text(text[b, :lengths_text[b], :])
            text_encodings.append(encoding)

        for b in range(batch_size):
            encoding = self.fc_cmd(commands[b, :lengths_cmd[b], :])
            cmd_encodings.append(encoding)

        text_encodings = pad_sequence(text_encodings, batch_first=True)
        cmd_encodings = pad_sequence(cmd_encodings, batch_first=True)
        text_encodings = text_encodings.sum(dim=1)
        cmd_encodings = cmd_encodings.sum(dim=1)

        text_encodings = torch.reshape(text_encodings, (batch_size, 8, 4, 4))
        cmd_encodings = torch.reshape(cmd_encodings, (batch_size, 8, 4, 4))
        combined_features = torch.cat([conv_out, text_encodings, cmd_encodings], 1)

        return self.final(combined_features)


class MultimodalRNNJEPS(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.feature_extractor_cnn = resnet50(pretrained=True) # Automatically load weights for ImageNet

        self.embedding = nn.Embedding(cfg.DATASET.VOCABULARY_SIZE, cfg.MRNN.EMBEDDING_DIM)
        self.fc_cmd = nn.Linear(cfg.DATASET.NUM_COMMANDS, cfg.MRNN.MULTIMODAL_LAYER_DIM)
        self.lstm_to_multimodal = nn.Linear(cfg.MRNN.HIDDEN_DIM, cfg.MRNN.MULTIMODAL_LAYER_DIM)
        self.cnn_to_multimodal = nn.Linear(self.feature_extractor_cnn.fc.in_features, cfg.MRNN.MULTIMODAL_LAYER_DIM)
        self.lstm = nn.LSTM(cfg.MRNN.EMBEDDING_DIM, cfg.MRNN.HIDDEN_DIM)
        self.scoring_text = nn.Linear(cfg.MRNN.MULTIMODAL_LAYER_DIM, cfg.DATASET.VOCABULARY_SIZE)
        self.scoring_cmd = nn.Linear(cfg.MRNN.MULTIMODAL_LAYER_DIM, cfg.DATASET.NUM_COMMANDS)

        # Freeze CNN so it will not be trained
        freeze_module(self.feature_extractor_cnn)

    def forward(self, per_img, goal_img, text, commands, lengths_text, lengths_cmd):
        batch_size = text.shape[0]

        text = self.embedding(text)

        _, feats_per = self.feature_extractor_cnn(per_img) # (batch_size, feat_dim)
        _, feats_goal = self.feature_extractor_cnn(goal_img)

        feats_per = self.cnn_to_multimodal(feats_per).unsqueeze(1)
        feats_goal = self.cnn_to_multimodal(feats_goal).unsqueeze(1)

        cmd_encodings = []

        for b in range(batch_size):
            encoding = self.fc_cmd(commands[b, :lengths_cmd[b], :])
            cmd_encodings.append(encoding)

        cmd_encodings = pad_sequence(cmd_encodings, batch_first=True)
        cmd_encodings = cmd_encodings.sum(dim=1).unsqueeze(1)

        lstm_outs = self.lstm(text)[0]
        multimodal = self.lstm_to_multimodal(lstm_outs)

        multimodal += cmd_encodings
        multimodal += feats_per
        multimodal += feats_goal

        scores_text = self.scoring_text(multimodal)
        scores_cmd = self.scoring_cmd(multimodal)

        return scores_text, scores_cmd


if __name__ == '__main__':
    from utils.parser import load_config
    from utils.data_utils import get_bacth
    cfg_file = load_config('configs/default.yaml')

    test_num = 2 # 2, 3, 5, 6, 15, 30

    if test_num % 2 == 0:
        aejeps = AutoencoderJEPS(cfg_file)
        batch = get_bacth(cfg_file)
        print("Testing AEJEPS in mode text")
        goal_image, lang, cmd = aejeps(*batch, mode='train')
        print(goal_image.shape, lang.shape, cmd.shape)

        print("Testing AEJEPS in mode text")
        goal_image, lang, cmd = aejeps(*batch, mode='text')
        print(goal_image.shape, lang.shape, cmd.shape)

        print("Testing AEJEPS in mode command")
        goal_image, lang, cmd = aejeps(*batch, mode='command')
        print(goal_image.shape, lang.shape, cmd.shape)

    if test_num % 3 == 0:
        compjeps = CompositeJEPS(cfg_file)
        per_image, goal_image, text, commands, lengths_text, lengths_cmd = get_bacth(cfg_file)

        outputs = compjeps(per_image, lengths_text, goal_img=goal_image, text=text)
        print([i.shape for i in outputs if i is not None])
        outputs = compjeps(per_image, lengths_cmd, goal_img=goal_image, commands=commands)
        print([i.shape for i in outputs if i is not None])
        outputs = compjeps(per_image, lengths_text, commands=commands, text=text, lengths_cmd=lengths_cmd, verbose=True)
        print([i.shape for i in outputs if i is not None])

        print("Testing Text generator of the composite model in mode train")
        outputs = compjeps.text_generator(per_image, goal_image, commands, text, lengths_cmd, mode='train')
        print(outputs.shape)

        print("Testing command generator of the composite model in mode train")
        outputs = compjeps.command_generator(per_image, goal_image, text, commands, lengths_text, mode='train')
        print(outputs.shape)

        generator = Generator(cfg_file)
        out = generator(torch.randn(4, 256, 1, 1), text, commands, lengths_text, lengths_cmd)
        print(out.shape)
        discriminator = Discriminator(cfg_file)

        out = discriminator(out, text, commands, lengths_text, lengths_cmd)
        print("Discriminator", out.shape)

    if test_num % 5 == 0:
        mrnn = MultimodalRNNJEPS(cfg_file)

        batch = get_bacth(cfg_file)
        mrnn(*batch)
