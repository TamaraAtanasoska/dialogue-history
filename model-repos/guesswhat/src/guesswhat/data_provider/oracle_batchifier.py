import numpy as np
import collections
from PIL import Image

from generic.data_provider.batchifier import AbstractBatchifier

from generic.data_provider.image_preprocessors import get_spatial_feat, resize_image
from generic.data_provider.nlp_utils import padder

answer_dict = \
    {'Yes': np.array([1, 0, 0], dtype=np.int32),
       'No': np.array([0, 1, 0], dtype=np.int32),
       'N/A': np.array([0, 0, 1], dtype=np.int32)
    }

class OracleBatchifier(AbstractBatchifier):

    def __init__(self, tokenizer, sources, status=list()):
        self.tokenizer = tokenizer
        self.sources = sources
        self.status = status

    def filter(self, games):
        if len(self.status) > 0:
            return [g for g in games if g.status in self.status]
        else:
            return games


    def apply(self, games):
        sources = self.sources
        tokenizer = self.tokenizer
        batch = collections.defaultdict(list)

        for i, game in enumerate(games):
            batch['raw'].append(game)

            image = game.image

            if 'question' in sources:
                assert  len(game.questions) == 1
                batch['question'].append(tokenizer.apply(game.questions[0]))

            if 'answer' in sources:
                assert len(game.answers) == 1
                batch['answer'].append(answer_dict[game.answers[0]])

            if 'category' in sources:
                batch['category'].append(game.object.category_id)

            if 'spatial' in sources:
                spat_feat = get_spatial_feat(game.object.bbox, image.width, image.height)
                batch['spatial'].append(spat_feat)

            if 'crop' in sources:
                batch['crop'].append(game.object.get_crop())

            if 'image' in sources:
                batch['image'].append(image.get_image())

            if 'mask' in sources:
                assert "image" in batch['image'], "mask input require the image source"
                mask = game.object.get_mask()
                ft_width, ft_height = batch['image'][-1].shape[1],\
                                     batch['image'][-1].shape[2] # Use the image feature size (not the original img size)
                mask = resize_image(Image.fromarray(mask), height=ft_height, width=ft_width)
                batch['mask'].append(mask)

        # pad the questions
        if 'question' in sources:
            batch['question'], batch['seq_length'] = padder(batch['question'], padding_symbol=tokenizer.word2i['<padding>'])

        return batch



