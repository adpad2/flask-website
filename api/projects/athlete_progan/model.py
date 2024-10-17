import torch
import torch.nn as nn
import torch.nn.functional as F

feature_factor_dict = {4: 1, 8: 1, 16: 1, 32: 1, 64: 1 / 2, 128: 1 / 4}


class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        """
        A 2D convolutional layer that scales the weights during runtime.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int): The kernel size for the convolutional layer.
            stride (int): The stride for the convolutional layer.
            gain(int): The normalization constant that controls the scaling of the weights.
        """

        super(WSConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5

        # Split the bias out of the convolutional layer - it will be added at the end to the output of the
        # convolutional layer.
        self.bias = self.conv.bias
        self.conv.bias = None

        # Initialize the convolutional layer.
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        Run a forward pass through the convolutional layer.

        Args:
            x (Tensor): The input tensor.

        Return:
            Tensor: The output from the convolutional layer.
        """

        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class PixelNorm(nn.Module):
    def __init__(self):
        """
        A block that normalizes the feature vector for every pixel-location to unit length.
        """

        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        """
        Normalize the input tensor such that the feature vector for every pixel-location has unit length.

        Args:
            x (Tensor): The input tensor.

        Return:
            Tensor: The normalized output tensor.
        """

        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixel_norm=True):
        """
        A block made up of 2 convolutional layers. Each convolutional layer process the input Tensor
        without changing the size of the feature maps (i.e. keeping the size of dimensions 2 and 3
        constant).

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            use_pixel_norm (boolean): Whether or not to apply PixelNorm after each convolutional layer.
        """

        super(ConvBlock, self).__init__()
        self.use_pn = use_pixel_norm
        self.pn = PixelNorm()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Run a forward pass through the 2 convolutional layers.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after running it through the convolutional layers
        """

        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x


class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, embedding_size, num_teams=40, num_builds=3, num_skin_tones=5,
                 img_channels=3):
        """
        A neural network that takes a vector as input and produces a (3 x N x N) image.

        Args:
            z_dim (int): The size of the random input vector.
            in_channels (int): The base number of channels which will be multiplied with the
                feature_factor_dict to determine the number of channels for a given layer.
            embedding_size (int): The size of each of the 3 embedding vectors.
            num_teams (int): The number of different team labels.
            num_builds (int): The number of different build labels.
            num_skin_tones (int): The number of different skin tone labels.
            img_channels (int): The number of channels in the final output image (e.g. 3 for RGB).
        """

        super(Generator, self).__init__()

        # A block that takes a tensor of shape (z_dim x 1 x 1) and converts it into a tensor of
        # shape (in_channels x 4 x 4).
        self.initial = nn.Sequential(
            # Shape: ((z_dim + 3 * embedding_size), 1, 1).
            # 3 embeddings will be concatenated with the input vector, so the size of the vecotr will
            # be (z_dim + 3 * embedding_size).
            PixelNorm(),
            # Shape: ((z_dim + 3 * embedding_size), 1, 1).
            nn.ConvTranspose2d(z_dim + 3 * embedding_size, in_channels, 4, 1, 0),
            # Shape: (in_channels, 4, 4).
            nn.LeakyReLU(0.2),
            # Shape: (in_channels, 4, 4).
            WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            # Shape: (in_channels, 4, 4).
            nn.LeakyReLU(0.2),
            # Shape: (in_channels, 4, 4).
            PixelNorm()
            # Shape: (in_channels, 4, 4).
        )

        # This is a dictionary of convolutional blocks that maps the image size to upscale to with
        # the block that upscales to that image size.
        self.conv_blocks = nn.ModuleDict({})
        # This is a dictionary of convolutional layers that maps the image size with the block that
        # converts a feature map of that size into a 3 channel (i.e. RGB) image.
        self.to_rgb_layers = nn.ModuleDict(
            {"4": WSConv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0)}
        )

        for img_size in feature_factor_dict.keys():
            # self.initial is used to upscale to a 4x4 image, so a seperate convolutional layer is not
            # necessary.
            if img_size == 4:
                continue

            # This is the number of channels the feature map of the previous image size has.
            conv_in_channels = int(in_channels * feature_factor_dict[img_size // 2])
            # This is the number of channels the feature map of the current image size has.
            conv_out_channels = int(in_channels * feature_factor_dict[img_size])

            self.conv_blocks[str(img_size)] = ConvBlock(conv_in_channels, conv_out_channels)
            self.to_rgb_layers[str(img_size)] = \
                WSConv2d(conv_out_channels, img_channels, kernel_size=1, stride=1, padding=0)

        # Create 3 embeddings for the 3 controlled variables. Team controls the jersey of the generated
        # players, build controls the weight/bulk, and skin tone controls the skin colour. These
        # embeddings will be concatenated to the random input vector.
        self.embed_team = nn.Embedding(num_teams, embedding_size)
        self.embed_build = nn.Embedding(num_builds, embedding_size)
        self.embed_skin_tone = nn.Embedding(num_skin_tones, embedding_size)

    def fade_in(self, alpha, upscaled, generated):
        """
        Fade the generated feature map in with the upscaled feature map. This ensures that the last
        convolutional block (which produces the generated feature map) starts by having a small
        impact on the output and eventually grows to have full control (as the value of alpha is
        increased to 1).

        Args:
            alpha (float): A value between 0 and 1 that controls the impact of the generated image
                (and inversely controls the impact of the upscaled image).
            upscaled (Tensor): The feature map which comes from the image size that is 2x
                smaller which is then upscaled by 2x.
            generated (Tensor): The feature map generated by the last convolutional block.

        Returns:
            Tensor: A feature map that is a conical combination of the generated and upscaled feature
                maps.
        """

        return alpha * generated + (1 - alpha) * upscaled

    def forward(self, x, team_labels, build_labels, skin_tone_labels, alpha=1, image_size=128):
        """
        Run a forward pass through the generator to produce RGB images.

        Args:
            x (Tensor): The random input vector.
            team_labels (Tensor): A vector containing the integer team labels.
            build_labels (Tensor): A vector containing the integer build labels.
            skin_tone_labels (Tensor): A vector containing the integer skin tone labels.
            alpha (float): A value between 0 and 1 that controls the impact of the generated image in
                the final convolutional block of the Generator (and inversely controls the impact of
                the upscaled image).
            image_size (int): The target image size of the generated images (must be a power of 2 in
                between 4 and 128).
        Returns:
            Tensor: The generated image with shape (3 x image_size x image_size).
        """

        # Embed the 3 controlled variables and unsqueeze so that they have shape (batch_size,
        # embedding_size, 1, 1).
        team_embedding = self.embed_team(team_labels).unsqueeze(2).unsqueeze(3)
        build_embedding = self.embed_build(build_labels).unsqueeze(2).unsqueeze(3)
        skin_tone_embedding = self.embed_skin_tone(skin_tone_labels).unsqueeze(2).unsqueeze(3)
        # Concatenate the random input vector and all 3 embeddings to create the new input vector.
        x = torch.cat([x, team_embedding, build_embedding, skin_tone_embedding], dim=1)

        # Pass the new input vector through the inital block that upscales it to a 4x4 feature map.
        out = self.initial(x)

        curr_image_size = 4
        # This loop upscales the current feature map until it reaches the target image size.
        while curr_image_size < image_size:
            # Multiply the current image size by two to refelct the new size after upscaling.
            curr_image_size *= 2
            # Upscale the feature map by 2x.
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            # Run the upscaled feature map through the appropriate convolutional layer.
            out = self.conv_blocks[str(curr_image_size)](upscaled)

        # At the end of the above while loop, the current image size should be equal to the target
        # image size.
        assert curr_image_size == image_size, (curr_image_size, image_size)

        # Run the final feature map through a RGB layer to reduce the number of channels to 3.
        rgb_out = self.to_rgb_layers[str(image_size)](out)
        if image_size == 4:
            # For a 4x4 image, no fading in is required (because there is no 2x smaller image to
            # upscale and fade in).
            return torch.tanh(rgb_out)
        else:
            # For any other image size, upscale the 2x smaller image and fade it in to the current
            # image size.
            rgb_upscaled = self.to_rgb_layers[str(image_size // 2)](upscaled)
            return torch.tanh(self.fade_in(alpha, rgb_upscaled, rgb_out))


class Critic(nn.Module):
    def __init__(self, in_channels, num_teams=40, num_builds=3, num_skin_tones=5, img_channels=3):
        """
        A neural network that determines the realness of a batch of images.

        Args:
            in_channels (int): The base number of channels which will be multiplied with the
                feature_factor_dict to determine the number of channels for a given layer.
            num_teams (int): The number of different team labels.
            num_builds (int): The number of different build labels.
            num_skin_tones (int): The number of different skin tone labels.
            img_channels (int): The number of channels in the final output image (e.g. 3 for RGB).
        """

        super(Critic, self).__init__()

        # This is a dictionary of convolutional blocks that maps the image size to downscale from with
        # the block that downscales that image size.
        self.conv_blocks = nn.ModuleDict({})
        # This is a dictionary of convolutional layers that maps the image size with the block that
        # converts a 3 channel (i.e. RGB) image of that size into a feature map.
        self.from_rgb_layers = nn.ModuleDict(
            {"4": WSConv2d(img_channels, in_channels, kernel_size=1, stride=1, padding=0)}
        )

        self.leaky = nn.LeakyReLU(0.2)

        for image_size in feature_factor_dict.keys():
            # self.final is used to process the 4x4 feature map, so a seperate convolutional layer is
            # not necessary.
            if image_size == 4:
                continue

            # This is the number of channels the feature map of the current image size has.
            conv_in_channels = int(in_channels * feature_factor_dict[image_size])
            # This is the number of channels the feature map of the next image size has.
            conv_out_channels = int(in_channels * feature_factor_dict[image_size // 2])

            self.conv_blocks[str(image_size)] = \
                ConvBlock(conv_in_channels, conv_out_channels, use_pixel_norm=False)
            self.from_rgb_layers[str(image_size)] = \
                WSConv2d(img_channels, conv_in_channels, kernel_size=1, stride=1, padding=0)

        # Average pooling is used to downscale a feature map by 2x.
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # This is the final block that operates on the 4x4 feature map
        self.final = nn.Sequential(
            # Shape: ((in_channels + 4), 4, 4).
            # +4 to in_channels because we concatenate from MiniBatch std, team embedding, build embedding, and
            # skin tone embedding.
            WSConv2d(in_channels + 4, in_channels, kernel_size=3, padding=1),
            # Shape: (in_channels, 4, 4).
            nn.LeakyReLU(0.2),
            # Shape: (in_channels, 4, 4).
            WSConv2d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            # Shape: (in_channels, 1, 1).
            nn.LeakyReLU(0.2),
            # Shape: (in_channels, 1, 1).
            WSConv2d(in_channels, 1, kernel_size=1, padding=0, stride=1)
            # Shape: (1, 1, 1).
        )

        # Create 3 embeddings for the 3 known variables. Team designates the expected jersey of the generated
        # players, build designates the weight/bulk, and skin tone designates the skin colour. These
        # embeddings will be concatenated to the feature map inputted into the final block.
        self.embed_team = nn.Embedding(num_teams, 4 * 4)
        self.embed_build = nn.Embedding(num_builds, 4 * 4)
        self.embed_skin_tone = nn.Embedding(num_skin_tones, 4 * 4)

    def fade_in(self, alpha, downscaled, out):
        """
        Fade the outputted feature map in with the downscaled feature map. This ensures that the last
        convolutional block (which produces the outputted feature map) starts by having a small
        impact on the output and eventually grows to have full control (as the value of alpha is
        increased to 1).

        Args:
            alpha (float): A value between 0 and 1 that controls the impact of the outputted image
                (and inversely controls the impact of the downscaled image).
            downscaled (Tensor): The feature map which comes from the image size that is 2x larger
                which is then downscaled by 2x.
            out (Tensor): The feature map outputted by the last convolutional block.

        Returns:
            Tensor: A feature map that is a conical combination of the outputted and downscaled
                feature maps.
        """

        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        """
        Take the standard deviation across the batch (across all channels and pixels) and repeat this
        value across a new channel. This gives the critic information about the variation of images in
        the batch.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor which contains a channel with the standard deviation statistics.
        """

        # Compute the standard deviation across the batch and repeat it to match the shape of the input
        # along dimensions 0, 2, and 3
        batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        # Concatenate the standard deviation feature map as a separate channe;
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, team_labels, build_labels, skin_tone_labels, alpha=1, image_size=128):
        """
        Run a forward pass through the critic to predict the realness of a batch of images.

        Args:
            x (Tensor): The input images with shape (N, 3, image_size, image_size).
            team_labels (Tensor): A vector containing the integer team labels.
            build_labels (Tensor): A vector containing the integer build labels.
            skin_tone_labels (Tensor): A vector containing the integer skin tone labels.
            alpha (float): A value between 0 and 1 that controls the impact of the output image in
                the first convolutional block of the Critic (and inversely controls the impact of
                the downscaled image).
            image_size (int): The image size of the input images (must be a power of 2 in
                between 4 and 128).
        Returns:
            Tensor: A 1-dimensional tensor containing the realness scores of every image in the batch.
        """

        # Convert the RGB image into a feature map.
        out = self.leaky(self.from_rgb_layers[str(image_size)](x))

        if image_size > 4:
            # If the image size is larger than 4x4, the first layer has to be faded in.
            downscaled = self.leaky(self.from_rgb_layers[str(image_size // 2)](self.avg_pool(x)))
            out = self.avg_pool(self.conv_blocks[str(image_size)](out))

            # Fade the downscaled feature map and the output feature map together.
            out = self.fade_in(alpha, downscaled, out)

        curr_image_size = image_size // 2
        # This loop downscales the current feature map until it is 4x4.
        while curr_image_size > 4:
            # Run the upscaled feature map through the appropriate convolutional layer.
            out = self.conv_blocks[str(curr_image_size)](out)
            # Downscale the feature map by 2x.
            out = self.avg_pool(out)
            # Divide the current image size by two to refelct the new size after upscaling.
            curr_image_size = curr_image_size // 2

        # Concatenate minibatch standard deviation as a 4x4 feature map.
        out = self.minibatch_std(out)
        # Concantenate the embeddings of the 3 controlled variables as 3 4x4 feature maps.
        team_embedding = self.embed_team(team_labels).view(team_labels.shape[0], 1, 4, 4)
        build_embedding = self.embed_build(build_labels).view(build_labels.shape[0], 1, 4, 4)
        skin_tone_embedding = self.embed_skin_tone(skin_tone_labels).view(skin_tone_labels.shape[0], 1, 4, 4)
        out = torch.cat([out, team_embedding, build_embedding, skin_tone_embedding], dim=1)

        # Run the final feature map through the final layer and reshape the output to be a 1-dimensional
        # vector with length equal to the batch size.
        return self.final(out).view(out.shape[0], -1)
