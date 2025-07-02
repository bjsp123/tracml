import typing as tp
import tracdap.rt.api as trac

import pandas as pd
import tensorflow as tf


class PandaModel(trac.TracModel):
    def define_parameters(self) -> tp.Dict[str, trac.ModelParameter]:

       return {}
    
    def define_inputs(self) -> tp.Dict[str, trac.ModelInputSchema]:
        return {}

    def define_outputs(self) -> tp.Dict[str, trac.ModelOutputSchema]:
        return {}


    def run_model(self, ctx: trac.TracContext):

        ctx.log().info("Panda model is running")

        ctx.log().info(f"This is pandamodel")

if __name__ == "__main__":
    import tracdap.rt.launch as launch
    launch.launch_model(PandaModel, "config/pandamodel.yaml", "config/sys_config.yaml")