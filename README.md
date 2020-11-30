## Handwriting Recognition Novelty

Primary repository for DARPA funded project relating to novelty in handwriting recognition.
Predictors will be the main content here.

### TODO

- Add: BanglaWriting scripts:
    + Bangla lines script w/ func to obtain lines given image and metadata (annotations) and option to save the line images
    + DataLoader for conversion of Bangla label files into tsv format.
        + Index files
- Debug: CRNN:
    + setup Tensorboard w/ pytorch and compare Sam's model vs my model.
        - Mine has consistently had worse performance than Sam's despite appearing to be the same model and able to load the same weights. Now, as of 2020-11-30, my version always predicts empty string when using Sam's config for training.
        - Use tensorboad to figure out the difference in wiring, if any.
    + setup legacy class that is Sam's model but interfaced w/ in API's expected way.
    + move newer CRNN class to itself, making a parent class or loader to go between the two.
- Add: Image representation modification: modular functions w/ optional saving
    + Code the module function to change text images into different reprs
    + add optional saving of the resulting images.
    + Add DataLoaders to allow for doing this during training or eval.
- Refactor: Get the guts of the transcriptor into `hwr_novelty/hwr_novelty/`
    + Put eval / assessment code into `hwr_novelty/hwr_novelty/eval/`
    + Make ANN transcriptor abstract
        - CharacterEncoder
        - fit()
        - predict()
    + Make the CRNN transcriptor class
    + Config classes for models to streamline args and config parsing
- Add: Data class and code in `hwr_novelty/experiments/data/`
    + DataLoader
    + Modular Config handler for each part of Data config. Handling labels wrt open set correctly
        - Label specific config, more than just the list of labels
        - Config classes for data to streamline args and config parsing
- Add: Modular YAML Config files for ease of experimentation
    + This avoid the issue of rewriting the same thing and attempts to allow reusable parts w/ the intention of write something only once and use multiple times.
    + separate into basic components at least: data, model, eval
    + add `refer:` for when relying on other part of config (e.g. model relying on labels defined in data as they the same w/ no differences in some occurrences.)
    + log at least at debug level, if not info, the resulting config file after parsing all sources of configuaration, including defaults, config files, and args.
