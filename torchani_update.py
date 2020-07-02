import torch
from torch import Tensor
import numpy
import torchani
from torchani import ANIModel
from torchani import nn
from torchani import neurochem
from torchani.utils import ChemicalSymbolsToInts as CSTI
import os
from typing import NamedTuple, Tuple, Optional

local = os.path.abspath(os.getcwd())

model_path = os.path.join(local,r"ani-model-zoo-master\resources\ani-2x_8x")

const = neurochem.Constants(os.path.join(model_path, 'rHCNOSFCl-5.1R_16-3.5A_a8-4.params'))
aev_computer = torchani.AEVComputer(Rcr=const.Rcr, Rca=const.Rca, EtaR=const.EtaR, ShfR=const.ShfR, 
                                    EtaA=const.EtaA, Zeta=const.Zeta, ShfA=const.ShfA, ShfZ=const.ShfZ,
                                   num_species = const.num_species)

EShifter = neurochem.load_sae(os.path.join(model_path, 'sae_linfit.dat'))

ANI2_Network = neurochem.load_model(const.species, os.path.join(model_path,r'train7\networks'))


class SpeciesEnergies(NamedTuple):
    species: Tensor
    energies: Tensor



def overforward(self, species_aev: Tuple[Tensor, Tensor],cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:
    species, aev = species_aev
    species_ = species.flatten()
    aev = aev.flatten(0, 1)

    output = aev.new_zeros(species_.shape)

    for i, (_, m) in enumerate(self.items()):
        mask = (species_ == i)
        midx = mask.nonzero().flatten()
        if midx.shape[0] > 0:
            input_ = aev.index_select(0, midx)
            output.masked_scatter_(mask, m(input_).flatten())
    output = output.view_as(species)
    return SpeciesEnergies(species, output)

def oversae(self, species):

        intercept = 0.0
        if self.fit_intercept:
            intercept = self.self_energies[-1]

        self_energies = self.self_energies[species]
        self_energies[species == torch.tensor(-1, device=species.device)] = torch.tensor(0, device=species.device, dtype=torch.double)
        return self_energies + intercept



funcType = type(ANI2_Network.forward)
ANI2_Network.forward = funcType(overforward, ANI2_Network)

funcType = type(EShifter.sae)
EShifter.sae = funcType(oversae, EShifter)

finished_network = nn.Sequential(aev_computer, ANI2_Network, EShifter)

species_to_tensor = CSTI(['H','O',"F", 'C'])

coordinates = torch.tensor([[[-1.0,0.0,0.0],
                             [0.0,0.0,0.0],
                             [1.0,0.0,0.0]]])
species = species_to_tensor(['O', 'C', 'O']).unsqueeze(0)

print(finished_network((species, coordinates)))





