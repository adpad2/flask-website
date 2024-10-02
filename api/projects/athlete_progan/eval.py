import torch
import random
import os
from scipy.stats import truncnorm

TEAMS = ['ARI', 'ARI-2', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN',
         'DET', 'GB', 'HOU', 'IND', 'JAX', 'JAX-2', 'KC', 'MIA', 'MIN', 'NE', 'NE-2',
         'NO', 'NYG', 'NYJ', 'OAK', 'PHI', 'PIT', 'SD', 'SD-2', 'SEA', 'SF', 'STL',
         'STL-2', 'STL-3', 'TB', 'TB-2', 'TEN', 'TEN-2', 'WAS']
TEAM_NAMES = ['Arizona Cardinals (1)', 'Arizona Cardinals (2)', 'Atlanta Falcons', 'Baltimore Ravens',
              'Buffalo Bills', 'Carolina Panthers', 'Chicago Bears', 'Cincinnati Bengals', 'Cleveland Browns',
              'Dallas Cowboys', 'Denver Broncos', 'Detroit Lions', 'Green Bay Packers', 'Houston Texans',
              'Indianapolis Colts', 'Jacksonville Jaguars (1)', 'Jacksonville Jaguars (2)', 'Kansas City Chiefs',
              'Miami Dolphins', 'Minnesota Vikings', 'New England Patriots (1)', 'New England Patriots (2)',
              'New Orleans Saints', 'New York Giants', 'New York Jets', 'Oakland Raiders', 'Philidelphia Eagles',
              'Pittsburgh Steelers', 'San Diego Chargers (1)', 'San Diego Chargers (2)', 'Seattle Seahawks',
              'San Francisco 49ers', 'St. Louis Rams (1)', 'St. Louis Rams (2)', 'St. Louis Rams (3)',
              'Tampa Bay Buccaneers (1)', 'Tampa Bay Buccaneers (2)', 'Tennessee Titans (1)', 'Tennessee Titans (2)',
              'Washington Redskins']
BUILDS = ['light', 'medium', 'heavy']
SKIN_TONES = ['very-dark', 'dark', 'neutral', 'fair', 'very-fair']

def gen_images(netG, team, skin_tone, build, num_images):
    bg_tile = 20
    upscale_factor = 2

    image_size = 128

    if team != 'any':
        team_idx = TEAMS.index(team)
        team_tensor = torch.tensor([team_idx] * num_images, dtype=torch.int)
    else:
        team_tensor = torch.tensor([random.randint(0, len(TEAMS) - 1) for i in range(num_images)], dtype=torch.int)

    if build != 'any':
        build_idx = BUILDS.index(build)
        build_tensor = torch.tensor([build_idx] * num_images, dtype=torch.int)
    else:
        build_tensor = torch.tensor([random.randint(0, len(BUILDS) - 1) for i in range(num_images)], dtype=torch.int)

    if skin_tone != 'any':
        skin_tone_idx = SKIN_TONES.index(skin_tone)
        skin_tone_tensor = torch.tensor([skin_tone_idx] * num_images, dtype=torch.int)
    else:
        skin_tone_tensor = torch.tensor([random.randint(0, len(SKIN_TONES) - 1) for i in range(num_images)], dtype=torch.int)

    flattened_noise = truncnorm.rvs(-1, 1, size=num_images * 32)
    noise = torch.tensor(flattened_noise, dtype=torch.float).view((num_images, 32, 1, 1))

    fake = netG(noise.detach(), team_tensor.detach(), build_tensor.detach(), skin_tone_tensor.detach(),
                alpha=1, image_size=image_size)
    # Crop 5 pixels from the top, 6 pixels from the bottom, and 14 pixels from either side. This removes the
    # black border and returns a 117x100 image.
    fake = fake[:, :, 5:-6, 14:-14]
    return fake
