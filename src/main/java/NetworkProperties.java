public enum NetworkProperties {
    INPUTS(30),
    HIDDEN_NODES(30),
    OUTPUTS(10),
    BATCH_SIZE(40),
    ITERATIONS(1000);

    public final int property;

    NetworkProperties(int property) {
        this.property = property;
    }
}
