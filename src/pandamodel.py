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
            trac.F("metric_name", trac.STRING, label="Metric Name"),
            trac.F("metric_value", trac.FLOAT, label="Metric Value"))
        
        approved_loans = trac.load_schema(schemas, "loans_schema_kaggle.csv")

        return {"loans_dq_metrics": dq_metrics,
                "approved_loans_clean": trac.ModelOutputSchema(approved_loans)}


    def run_model(self, ctx: trac.TracContext):

        ctx.log().info("Panda model is running")

        ctx.log().info(f"This is pandamodel")

        loansdf = ctx.get_pandas_table("approved_loans")

        dqdf = pd.DataFrame({"metric_name":["metric a", "metric b"], "metric_value":[0.9,0.93]})

        #right, first output some dummy DQ metrics
        ctx.put_pandas_table("loans_dq_metrics", dqdf)

        #now output the 'cleaned' data but don't do any actual cleaning as such
        ctx.put_pandas_table("approved_loans_clean", loansdf)


if __name__ == "__main__":
    import tracdap.rt.launch as launch
    launch.launch_model(PandaModel, "config/pandamodel.yaml", "config/sys_config.yaml")