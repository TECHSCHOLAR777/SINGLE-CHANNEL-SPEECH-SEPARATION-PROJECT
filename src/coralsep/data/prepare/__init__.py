"""Corpus preparation. One module per external dataset.

Each downloads or converts a public corpus into the 8 kHz layout the mixer
expects, and verifies the result rather than assuming it. None of them is
imported at training time; they are run once, ahead of everything else.
"""
