import traits.api as traits
from soma import aims
from capsul.api import Process
from label_resample import resample


class LabelResample(Process):
    '''
        Resample a volume that as discret values
    '''

    def __init__(self):
        super(LabelResample, self).__init__()

        self.add_trait('input_image', traits.File(
            output=False, desc='Labelled image to transform'))
        self.add_trait('transformation', traits.File(
            output=False, optional=True, desc='Transformation file .trm'))
        self.add_trait('sx', traits.Float(-1, output=False,
                                        desc='Output resolution (X axis)'))
        self.add_trait('sy', traits.Float(-1, output=False,
                                        desc='Output resolution (Y axis)'))
        self.add_trait('sz', traits.Float(-1, output=False,
                                        desc='Output resolution (Z axis)'))
        self.add_trait('background', traits.Int(0, output=False,
                                        desc='Background value/label'))
        self.add_trait('output_image', traits.File(
            output=False, desc='file (.json) storing the hyperparameters'
                               ' (cutting threshold)'))

    def _run_process(self):
        # Read inputs
        resampled = resample(self.input_image, self.transformation,
                             (self.sx, self.sy, self.sz), self.background)
        aims.write(resampled, self.output_image)
