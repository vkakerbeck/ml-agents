import tensorflow as tf

class Teacher(object):
    #adapted from https://leimao.github.io/blog/Save-Load-Inference-From-TF-Frozen-Graph/
    def __init__(self, model_filepath):

        # The file path of model
        self.model_filepath = model_filepath
        # Initialize the model
        self.load_graph(model_filepath = self.model_filepath)

    def load_graph(self, model_filepath):
        '''
        Lode trained model.
        '''
        print('Loading model...')
        self.graph = tf.Graph()

        with tf.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        print('Check out the input placeholders:')
        nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
        for node in nodes:
            print(node)

        with self.graph.as_default():
            self.VisInput = tf.placeholder(shape=[None, 168, 168, 3], dtype=tf.float32,name="visual_observation_0")
            self.VecInput = tf.placeholder(shape=[None, 8], dtype=tf.float32,name='vector_observation')
            self.action_masks = tf.placeholder(shape=[None, 11], dtype=tf.float32, name="action_masks")
            tf.import_graph_def(graph_def, {'visual_observation_0': self.VisInput,
                                            'vector_observation': self.VecInput,
                                            'action_masks':self.action_masks})
            #self.init = tf.global_variables_initializer()

        self.graph.finalize()

        print('Model loading complete!')

        # Get layer names
        layers = [op.name for op in self.graph.get_operations()]
        for layer in layers:
            print(layer)

        """
        # Check out the weights of the nodes
        weight_nodes = [n for n in graph_def.node if n.op == 'Const']
        for n in weight_nodes:
            print("Name of the node - %s" % n.name)
            # print("Value - " )
            # print(tensor_util.MakeNdarray(n.attr['value'].tensor))
        """

        # In this version, tf.InteractiveSession and tf.Session could be used interchangeably.
        #self.sess = tf.InteractiveSession(graph = self.graph)
        self.sess = tf.Session(graph = self.graph)

        #self.sess.run(self.init)

    def get_teacher_action(self, visIn, vecIn, ActMask):
        #'dense/kernel:0', 'dense_1/kernel:0', 'dense_2/kernel:0', 'dense_3/kernel:0', 'dense_4/kernel:0'

        # Know your output node name
        output_tensor = self.graph.get_tensor_by_name("import/action:0")
        #enc_tensor = self.graph.get_tensor_by_name('import/concat:0')
        #output = self.sess.run([output_tensor, enc_tensor], feed_dict = {self.VisInput: visIn, self.VecInput: vecIn, self.action_masks: ActMask})
        output = self.sess.run(output_tensor, feed_dict = {self.VisInput: visIn, self.VecInput: vecIn, self.action_masks: ActMask})

        return output
