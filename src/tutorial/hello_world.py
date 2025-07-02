import typing as tp
import tracdap.rt.api as trac


class HelloWorldModel(trac.TracModel):
    def define_parameters(self) -> tp.Dict[str, trac.ModelParameter]:

        return trac.define_parameters(
            trac.P(
                "meaning_of_life", trac.INTEGER,
                label="The answer to the ultimate question of life, the universe and everything"))
    
    def define_inputs(self) -> tp.Dict[str, trac.ModelInputSchema]:
        return {}

    def define_outputs(self) -> tp.Dict[str, trac.ModelOutputSchema]:
        return {}


    def run_model(self, ctx: trac.TracContext):

        ctx.log().info("Hello world model is running")

        meaning_of_life = ctx.get_parameter("meaning_of_life")
        ctx.log().info(f"The meaning of life is {meaning_of_life}")

if __name__ == "__main__":
    import tracdap.rt.launch as launch
    launch.launch_model(HelloWorldModel, "config/hello_world.yaml", "config/sys_config.yaml")