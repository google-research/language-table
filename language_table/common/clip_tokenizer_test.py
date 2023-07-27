# coding=utf-8
# Copyright 2024 The Language Tale Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for clip_tokenizer."""

from clip.simple_tokenizer import default_bpe
from clip.simple_tokenizer import SimpleTokenizer

from language_table.common import clip_tokenizer

import numpy as np
import tensorflow as tf


class ClipTokenizerTest(tf.test.TestCase):

  def test_matches_simple_tokenizer(self):
    vocab_lookup = clip_tokenizer.create_vocab(bpe_path=default_bpe())
    tokenizer = clip_tokenizer.ClipTokenizer(vocab_lookup)

    simple_tokenizer = SimpleTokenizer(default_bpe())

    all_instructions = [
        "pull the red moon apart from the star, red pentagon, and blue moon",
        "hexagon cube star crescent moon block",
        "I must've gone, to where you're going and place where he'll go,",
        "push the blue hexagon to the left!!!",
        " a    thing with   lots of spaces",
        "a picture of an elephantcat ",
        " a picture   of an \n elephantcat  ",
        "A PICTURE of AN elephantCAT",

        # Current known failure cases.
        # "&lt;html&gt;I want to escape HTML &lt;/html&gt;"
    ]

    for instruction in all_instructions:
      result = clip_tokenizer.tokenize_text(instruction, tokenizer)
      simple_result = _simple_tokenize(simple_tokenizer, [instruction])
      np.testing.assert_equal(result, simple_result, instruction)


def _simple_tokenize(tokenizer, texts, context_length = 77):
  sot_token = tokenizer.encoder["<|startoftext|>"]
  eot_token = tokenizer.encoder["<|endoftext|>"]
  all_tokens = [
      [sot_token] + tokenizer.encode(text) + [eot_token] for text in texts
  ]
  result = np.zeros((len(all_tokens), context_length), dtype=np.compat.long)

  for i, tokens in enumerate(all_tokens):
    if len(tokens) > context_length:
      raise RuntimeError(
          f"Input {texts[i]} is too long for context length {context_length}")
    result[i, :len(tokens)] = np.asarray(tokens)

  return result


if __name__ == "__main__":
  tf.test.main()
