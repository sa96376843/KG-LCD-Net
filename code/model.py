# -*- coding: utf-8 -*-
"""
KG-LCD-Net: Knowledge-Graph-guided Multi-Layer Local-Community-Detection
network for retinal vessel segmentation.

Architecture (exactly as used in all experiments):
  * 4-scale compact residual U-Net, base width 24, 4-channel input
    (RGB fundus + Frangi vesselness);
  * Knowledge-graph prior module: 12-entity / 6-relation retinal vascular
    anatomy KG -> relational GCN -> attention pooling -> FiLM modulation
    of the bottleneck.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Retinal vascular anatomy knowledge graph                                   #
# --------------------------------------------------------------------------- #
KG_ENTITIES = ['OpticDisc', 'Macula', 'Fovea', 'SupArcade', 'InfArcade',
               'MainTrunk', 'Artery', 'Vein', 'Bifurcation', 'Crossing',
               'ThinVessel', 'BackgroundRegion']
KG_RELATIONS = ['partOf', 'connectedTo', 'branchesInto', 'crosses',
                'surroundedBy', 'adjacentTo']

KG_TRIPLES = [
    ('MainTrunk',   'branchesInto', 'Artery'),
    ('MainTrunk',   'branchesInto', 'Vein'),
    ('Artery',      'branchesInto', 'ThinVessel'),
    ('Vein',        'branchesInto', 'ThinVessel'),
    ('Artery',      'connectedTo',  'Bifurcation'),
    ('Vein',        'connectedTo',  'Bifurcation'),
    ('Bifurcation', 'branchesInto', 'ThinVessel'),
    ('Artery',      'crosses',      'Vein'),
    ('Crossing',    'partOf',       'Artery'),
    ('Crossing',    'partOf',       'Vein'),
    ('SupArcade',   'partOf',       'MainTrunk'),
    ('InfArcade',   'partOf',       'MainTrunk'),
    ('SupArcade',   'surroundedBy', 'BackgroundRegion'),
    ('InfArcade',   'surroundedBy', 'BackgroundRegion'),
    ('Macula',      'adjacentTo',   'SupArcade'),
    ('Macula',      'adjacentTo',   'InfArcade'),
    ('Fovea',       'partOf',       'Macula'),
    ('OpticDisc',   'connectedTo',  'MainTrunk'),
    ('ThinVessel',  'surroundedBy', 'BackgroundRegion'),
    ('Macula',      'surroundedBy', 'BackgroundRegion'),
]


def build_kg_adj():
    """Relation-specific adjacency A_r (E x E), symmetrised, self-loops."""
    E, R = len(KG_ENTITIES), len(KG_RELATIONS)
    e2i = {e: i for i, e in enumerate(KG_ENTITIES)}
    r2i = {r: i for i, r in enumerate(KG_RELATIONS)}
    A = torch.zeros(R, E, E)
    for h, r, t in KG_TRIPLES:
        i, j = e2i[h], e2i[t]
        A[r2i[r], i, j] = 1.0
        A[r2i[r], j, i] = 1.0
    A += torch.eye(E).unsqueeze(0)
    return A


# --------------------------------------------------------------------------- #
#  Knowledge-graph prior module                                               #
# --------------------------------------------------------------------------- #
class KGPrior(nn.Module):
    """R-GCN propagation + entity attention pooling -> context vector."""

    def __init__(self, d=32, dout=192):
        super().__init__()
        E, R = len(KG_ENTITIES), len(KG_RELATIONS)
        self.E = nn.Parameter(torch.randn(E, d) * 0.1)
        self.register_buffer('A', build_kg_adj())
        self.Wr = nn.Parameter(torch.randn(R, d, d) * 0.1)
        self.W0 = nn.Parameter(torch.randn(d, d) * 0.1)
        self.q = nn.Parameter(torch.randn(d) * 0.1)
        self.proj = nn.Sequential(nn.Linear(d, dout), nn.ReLU(),
                                  nn.Linear(dout, dout))

    def forward(self, alpha):
        """alpha: (B, E) image-adaptive entity activations."""
        H = self.E
        deg = self.A.sum(-1, keepdim=True).clamp(min=1.)
        An = self.A / deg
        msg = torch.einsum('rij,jd->rid', An, H)
        agg = torch.einsum('red,rdf->ref', msg, self.Wr).sum(0)
        H2 = torch.relu(agg + H @ self.W0)
        att = torch.softmax(H2 @ self.q, dim=0)
        ctx = torch.einsum('be,ed->bd', alpha * att.unsqueeze(0), H2)
        return self.proj(ctx)


# --------------------------------------------------------------------------- #
#  Compact residual U-Net                                                     #
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.c1 = nn.Conv2d(ci, co, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(co)
        self.c2 = nn.Conv2d(co, co, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(co)
        self.skip = (nn.Conv2d(ci, co, 1, bias=False)
                     if ci != co else nn.Identity())

    def forward(self, x):
        idt = self.skip(x)
        x = F.relu(self.b1(self.c1(x)), inplace=True)
        x = self.b2(self.c2(x))
        return F.relu(x + idt, inplace=True)


class KGLCDUNet(nn.Module):
    def __init__(self, cin=4, base=24, use_kg=True):
        super().__init__()
        b = base
        self.use_kg = use_kg
        self.e1 = ConvBlock(cin, b)
        self.e2 = ConvBlock(b, b * 2)
        self.e3 = ConvBlock(b * 2, b * 4)
        self.e4 = ConvBlock(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        if use_kg:
            self.ent_proj = nn.Linear(b * 8, len(KG_ENTITIES))
            self.kg = KGPrior(d=32, dout=b * 8)
            self.film = nn.Linear(b * 8, 2 * b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.d3 = ConvBlock(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.d2 = ConvBlock(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.d1 = ConvBlock(b * 2, b)
        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        bn = self.e4(self.pool(e3))
        if self.use_kg:
            gap = bn.mean(dim=(2, 3))
            alpha = torch.softmax(self.ent_proj(gap), dim=1)
            ctx = self.kg(alpha)
            gamma, beta = self.film(ctx).chunk(2, dim=1)
            bn = bn * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        d3 = self.d3(torch.cat([self.up3(bn), e3], 1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    m = KGLCDUNet(use_kg=True)
    print('full params:', count_params(m))
    print('backbone params:', count_params(KGLCDUNet(use_kg=False)))
    print('out shape:', m(torch.randn(2, 4, 96, 96)).shape)
