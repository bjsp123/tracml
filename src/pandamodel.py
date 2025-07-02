import typing as tp
import tracdap.rt.api as trac

import pandas as pd
import tensorflow as tf

import schemas as schemas


class PandaModel(trac.TracModel):
    def define_parameters(self) -> tp.Dict[str, trac.ModelParameter]:

       return {}
    
    def define_inputs(self) -> tp.Dict[str, trac.ModelInputSchema]:
        approved_loans = trac.load_schema(schemas, "loans_schema_kaggle.csv")

        return {"approved_loans": trac.ModelInputSchema(approved_loans)}

    def define_outputs(self) -> tp.Dict[str, trac.ModelOutputSchema]:
        dq_metrics = trac.define_output_table(
            trac.F("metric1", trac.FLOAT, label="Metric 1"),
            trac.F("metric2", trac.FLOAT, label="Metric 2"))

        return {"loans_dq_metrics": dq_metrics}


    def run_model(self, ctx: trac.TracContext):

        ctx.log().info("Panda model is running")

        ctx.log().info(f"This is pandamodel")

if __name__ == "__main__":
    import tracdap.rt.launch as launch
    launch.launch_model(PandaModel, "config/pandamodel.yaml", "config/sys_config.yaml")