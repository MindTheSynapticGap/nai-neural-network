public enum NetworkProperties {
    INPUTS(30),
    HIDDEN_NODES(30),
    OUTPUTS(3),
    BATCH_SIZE(10),
    ITERATIONS(1000);

    public final int property;

    NetworkProperties(int property) {
        this.property = property;
    }
}
