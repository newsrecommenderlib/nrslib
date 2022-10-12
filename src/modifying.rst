Adding customised contents
=====================================================

Adding and preparing other datasources
----------------
The data processing steps vary depending on the used model.

Neural Collaborative Filtering, MKR, RippleNet
+++++++++++++++++++
The Neural Collaborative Filtering model depends exclusively on rating information, MKR and RippleNet also amke use of knowledge graph information.
The rating information needs to be extracted from the source data prior to model training.
The training rating data needs to include the user id, the news article id and the rating.

=======    =======    ======
User id    News id    Rating
=======    =======    ======
   0        1321         1
   0         23          0
   0         59          0
   1         789         1
=======    =======    ======

The validation and test rating data needs to include the user id, the news article id, the rating and the impression id that a rating belongs to.

=======    =======    ======    ======
User id    News id    Rating     Impr.
=======    =======    ======    ======
   0        1321        1         0
   0        23          0         0
   0        590         0         47
   1        7890        1         987
=======    =======    ======    ======

This data in numpy format can then be instantiated as a ``RatingsDataset`` object which will be loaded into the PyTorch Lightning Dataloader, ready for usage in the models.

.. autoclass:: src.datamodules.mind.dataset.RatingsDataset
   :members:


The models MKR and RippleNet make use of knowledge graphs in addition to the implicit rating information of users.
The knowledge graphs may consist of any side information associated to news articles. In case of the MIND dataset this side information consists of categories, subcategories, article headlines, article abstracts and wikidata information associated with these article entities.

Knowledge graphs for MKR must follow the following structure, where the head represents the news id and the tail represents the entity id.

======    ========    ======
 Head     Relation     Tail
======    ========    ======
  0          0        20123
  0          1        10198
  0          2        38976
  1          0        20133
======    ========    ======

For the MKR model the knowledge graph also gets loaded into the model via Dataloaders and thus gets instantiated as a ``KGDataset`` object.

.. autoclass:: src.datamodules.mind.dataset.KGDataset
   :members:


Knowledge graphs for RippleNet also need the mirrored version of every relation.

======    ========    ======
 Head     Relation     Tail
======    ========    ======
  0          0        20123
20123        1          0
  0          2        10198
10198        3          0
======    ========    ======

For the RippleNet model the knowledge graph does not get loaded via Dataloaders but gets automatically loaded into the model during instantiation.





ToDo: other models data


How to create your own models, datamodules
----------------
All MIND datamodules inherit from the class ``MINDDataModule`` which unites all model-independent instantiation parameters.


.. autoclass:: src.datamodules.mind.datamodule_Base.MINDDataModule
   :members:





